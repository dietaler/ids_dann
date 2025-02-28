import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import mnist
import mnistm
from utils import save_model
from utils import visualize
from utils import set_model_mode
import params
import cicids2017
import cicids2018

# Source : 0, Target :1
# source_test_loader = mnist.mnist_test_loader
# target_test_loader = mnistm.mnistm_test_loader
source_test_loader = cicids2017.test_loader_2017
target_test_loader = cicids2018.test_loader_2018

def source_only(encoder, classifier, source_train_loader, target_train_loader):
    print("Training with only the source dataset")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=params.lr, momentum=0.9)

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            # print(source_image.shape)
            p = float(batch_idx + start_steps) / total_steps

            # source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            #debug
            # print(f"Before encoding: {source_image.shape}")  # 預期 [32, 3, 224, 224]
            # source_feature = encoder(source_image)
            # print(f"After encoding: {source_feature.shape}")  # 應該是 [32, feature_dim]

            # class_pred = classifier(source_feature)
            # print(f"Classifier output: {class_pred.shape}")  # 應該是 [32, num_classes]

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                total_processed = batch_idx * len(source_image)
                total_dataset = len(source_train_loader.dataset)
                percentage_completed = 100. * batch_idx / len(source_train_loader)
                # print(f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tClassification Loss: {class_loss.item():.4f}')
        test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='Source_only')

    save_model(encoder, classifier, None, 'Source-only')
    # visualize(encoder, 'Source-only')


def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader):
    print("Training with the DANN adaptation method")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=params.lr,
        momentum=0.9)

    for epoch in range(params.epochs_DANN):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs_DANN * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2 * 2. / (1. + np.exp(-10 * p)) - 1

            # source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss (只在偶數 epoch 訓練)
            if epoch % 2 == 0:
                domain_pred = discriminator(combined_feature, alpha)
                domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
                domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
                domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
                domain_loss = discriminator_criterion(domain_pred, domain_combined_label)
            else:
                domain_loss = torch.tensor(0.0).cuda()  # 在奇數 epoch 不計算 domain loss

            total_loss = class_loss + 0.1 * domain_loss
            total_loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % 100 == 0:
            #     print('[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tClassification Loss: {:.4f}\tDomain Loss: {:.4f}'.format(
            #         batch_idx * len(target_image), len(target_train_loader.dataset), 100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))

        test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='DANN')

    save_model(encoder, classifier, discriminator, 'DANN')
    # visualize(encoder, 'DANN')
