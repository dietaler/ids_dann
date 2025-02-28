import torch
import train
import mnist
import mnistm
import model
import cicids2017
import cicids2018

def main():
    # source_train_loader = mnist.mnist_train_loader
    # target_train_loader = mnistm.mnistm_train_loader
    source_train_loader = cicids2017.train_loader_2017
    target_train_loader = cicids2018.train_loader_2018

    if torch.cuda.is_available():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()

        train.source_only(encoder, classifier, source_train_loader, target_train_loader)
        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader)
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
