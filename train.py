# The testing module requires faiss
# So if you don't have that, then this import will break
from pytorch_metric_learning import losses, miners, samplers, trainers, testers, utils
import torch.nn as nn
import record_keeper
import pytorch_metric_learning.utils.logging_presets as logging_presets
from torchvision import datasets, models, transforms
import torchvision
import logging
logging.getLogger().setLevel(logging.INFO)
import os

import pytorch_metric_learning
from pytorch_metric_learning.testers.base_tester import BaseTester

logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)

from sklearn.metrics import accuracy_score
#from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
import pickle

import hydra
from omegaconf import DictConfig


# reprodcibile
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_experiment_name(cfg, extra_str):
    augmentation_text = "rotation_"+str(cfg.transform.transform_random_rotation_degrees) if cfg.transform.transform_random_rotation else ""
    extra_str+=augmentation_text
    experiment_name = "%s_model_%s_cl_%s_ml_%s_miner_%s_mix_ml_%02.2f_mix_cl_%02.2f_resize_%d_emb_size_%d_class_size_%d_opt_%s_lr_%02.2f_%s"%(cfg.dataset.name,
                                                                                                  cfg.model.model_name, 
                                                                                                  "cross_entropy", 
                                                                                                  cfg.embedder_loss.name, 
                                                                                                  cfg.miner.name, 
                                                                                                  cfg.loss.metric_loss, 
                                                                                                  cfg.loss.classifier_loss,
                                                                                                  cfg.transform.transform_resize,
                                                                                                  cfg.embedder.size,
                                                                                                  cfg.embedder.class_out_size,
                                                                                                  cfg.optimizer.name,
                                                                                                  cfg.optimizer.lr,
                                                                                                  extra_str
                                                                                                  #cfg.optimizer.momentum,
                                                                                                  #cfg.optimizer.weight_decay
                                                                                                  )
    return experiment_name


class OneShotTester(BaseTester):

    def __init__(self, end_of_testing_hook=None):
        super().__init__()
        self.max_accuracy = 0.0
        self.embedding_filename = ""
        self.end_of_testing_hook = end_of_testing_hook


    def __get_correct(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    #             print(correct)
        return correct


    def __accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = self.__get_correct(output, target, topk)
            batch_size = target.size(0)
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        #print(embeddings_and_labels)
        query_embeddings = embeddings_and_labels["val"][0]
        query_labels = embeddings_and_labels["val"][1]
        reference_embeddings = embeddings_and_labels["samples"][0]
        reference_labels = embeddings_and_labels["samples"][1]
        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings, query_embeddings, 1, False)
        knn_labels = reference_labels[knn_indices][:,0]

        accuracy = accuracy_score(knn_labels, query_labels)
        print(accuracy)
        with open(self.embedding_filename+"_last", 'wb') as f:
            print("Dumping embeddings for new max_acc to file", self.embedding_filename+"_last")
            pickle.dump([query_embeddings, query_labels, reference_embeddings, reference_labels, accuracy], f)
        accuracies["accuracy"] = accuracy
        keyname = self.accuracies_keyname("mean_average_precision_at_r") # accuracy as keyname not working
        accuracies[keyname] = accuracy


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_datasets(data_dir, cfg, mode="train"):

    common_transforms = []
    train_transforms = []
    test_transforms = []
    #if cfg.transform.transform_resize_match:
    common_transforms.append(transforms.Resize((cfg.transform.transform_resize,cfg.transform.transform_resize)))
    #else:
    #    common_transforms.append(transforms.Resize(cfg.transform.transform_resize))
    
    if cfg.transform.transform_random_resized_crop:
        train_transforms.append(transforms.RandomResizedCrop(cfg.transform.transform_resize))
    if cfg.transform.transform_random_horizontal_flip:
        train_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    if cfg.transform.transform_random_rotation:
        train_transforms.append(transforms.RandomRotation(cfg.transform.transform_random_rotation_degrees))#, fill=255))
    if cfg.transform.transform_random_shear:
        train_transforms.append(torchvision.transforms.RandomAffine(0,
                                                                    shear=(
                                                                        cfg.transform.transform_random_shear_x1,
                                                                        cfg.transform.transform_random_shear_x2,
                                                                        cfg.transform.transform_random_shear_y1,
                                                                        cfg.transform.transform_random_shear_y2
                                                                        ),
                                                                    fillcolor=255)) 
    if cfg.transform.transform_random_perspective:
        train_transforms.append(transforms.RandomPerspective(distortion_scale=cfg.transform.transform_perspective_scale, 
                                     p=0.5, 
                                     interpolation=3)
                                )
    if cfg.transform.transform_random_affine:
        train_transforms.append(transforms.RandomAffine(degrees=(cfg.transform.transform_degrees_min,
                                                                 cfg.transform.transform_degrees_max),
                                                        translate=(cfg.transform.transform_translate_a,
                                                                   cfg.transform.transform_translate_b),
                                                        fillcolor=255))
    data_transforms = {
            'train': transforms.Compose(common_transforms+train_transforms+[transforms.ToTensor()]),
            'test': transforms.Compose(common_transforms+[transforms.ToTensor()]),
            }

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),
            data_transforms["train"])





    # for the final model we can join train, validation, validation samples datasets
    print(mode)
    if mode == "final_train":
        #train_dataset = torch.utils.data.ConcatDataset([train_dataset,
        #        val_dataset,
        #        val_samples_dataset])

        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"),
                data_transforms["test"])

        samples_dataset = datasets.ImageFolder(os.path.join(data_dir, "samples"),
                data_transforms["test"])
        return train_dataset, test_dataset, samples_dataset
    else:
        if mode == "train":
            val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"),
                    data_transforms["test"])

            val_samples_dataset = datasets.ImageFolder(os.path.join(data_dir, "val_samples"),
                    data_transforms["test"])
            return train_dataset, val_dataset, val_samples_dataset

        if mode == "test":
            return train_dataset, test_dataset, samples_dataset


@hydra.main(config_path="config/config.yaml")
def train_app(cfg):
    print(cfg.pretty())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set trunk model and replace the softmax layer with an identity function
    trunk = torchvision.models.__dict__[cfg.model.model_name](pretrained=cfg.model.pretrained)
    
    #resnet18(pretrained=True)
    #trunk = models.alexnet(pretrained=True)
    #trunk = models.resnet50(pretrained=True)
    #trunk = models.resnet152(pretrained=True)
    #trunk = models.wide_resnet50_2(pretrained=True)
    #trunk = EfficientNet.from_pretrained('efficientnet-b2')
    trunk_output_size = trunk.fc.in_features
    trunk.fc = Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))

    embedder = torch.nn.DataParallel(MLP([trunk_output_size, cfg.embedder.size]).to(device))
    classifier = torch.nn.DataParallel(MLP([cfg.embedder.size, cfg.embedder.class_out_size])).to(device)

    # Set optimizers
    if cfg.optimizer.name == "sdg":
        trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "rmsprop":
        trunk_optimizer = torch.optim.RMSprop(trunk.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)



    # Set the datasets
    data_dir = os.environ["DATASET_FOLDER"]+"/"+cfg.dataset.data_dir
    print("Data dir: "+data_dir)

    train_dataset, val_dataset, val_samples_dataset = get_datasets(data_dir, cfg, mode=cfg.mode.type)
    print("Trainset: ",len(train_dataset), "Testset: ",len(val_dataset), "Samplesset: ",len(val_samples_dataset))

    # Set the loss function
    if cfg.embedder_loss.name == "margin_loss":
        loss = losses.MarginLoss(margin=cfg.embedder_loss.margin,nu=cfg.embedder_loss.nu,beta=cfg.embedder_loss.beta)
    if cfg.embedder_loss.name == "triplet_margin":
        loss = losses.TripletMarginLoss(margin=cfg.embedder_loss.margin)
    if cfg.embedder_loss.name == "multi_similarity":
        loss = losses.MultiSimilarityLoss(alpha=cfg.embedder_loss.alpha, beta=cfg.embedder_loss.beta, base=cfg.embedder_loss.base)
    if cfg.embedder_loss.name == "proxy_anchor":
        loss = losses.ProxyAnchorLoss(num_classes=100, embedding_size=cfg.embedder.size, margin=cfg.embedder_loss.margin, alpha=cfg.embedder_loss.alpha)
        #loss = losses.ProxyAnchorLoss(num_classes=cfg.embedder.class_out_size, embedding_size=cfg.embedder.size, margin=cfg.embedder_loss.margin, alpha=cfg.embedder_loss.alpha)
        loss = loss.to(device)
    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function

    if cfg.miner.name == "triplet_margin":
        #miner = miners.TripletMarginMiner(margin=0.2)
        miner = miners.TripletMarginMiner(margin=cfg.miner.margin)
    if cfg.miner.name == "multi_similarity":
        miner = miners.MultiSimilarityMiner(epsilon=cfg.miner.epsilon)
        #miner = miners.MultiSimilarityMiner(epsilon=0.05)

    #loss = losses.CrossBatchMemory(loss, cfg.embedder.size, memory_size=1024, miner=miner) 
    #extra_str = "cb_mem"
    extra_str = ""

    batch_size = cfg.trainer.batch_size
    num_epochs = cfg.trainer.num_epochs
    iterations_per_epoch = cfg.trainer.iterations_per_epoch
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))
    


    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": cfg.loss.metric_loss, "classifier_loss": cfg.loss.classifier_loss}


    schedulers = {
            #"metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            }


    experiment_name = get_experiment_name(cfg, extra_str)
    print(experiment_name)

    record_keeper, _, _ = logging_presets.get_record_keeper("logs/%s"%(experiment_name), "tensorboard/%s"%(experiment_name))
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset}
    model_folder = "example_saved_models/%s/"%(experiment_name)


    # Create the tester
    tester = OneShotTester(
            end_of_testing_hook=hooks.end_of_testing_hook, 
            #size_of_tsne=20
            )
    #tester.embedding_filename=data_dir+"/embeddings_pretrained_triplet_loss_multi_similarity_miner.pkl"
    tester.embedding_filename=data_dir+"/"+experiment_name+".pkl"
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
    trainer = trainers.TrainWithClassifier(models,
            optimizers,
            batch_size,
            loss_funcs,
            mining_funcs,
            train_dataset,
            sampler=sampler,
            lr_schedulers=schedulers,
            dataloader_num_workers = cfg.trainer.batch_size,
            loss_weights=loss_weights,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook
            )

    trainer.train(num_epochs=num_epochs)

    tester = OneShotTester()

if __name__ == "__main__":
    train_app()
