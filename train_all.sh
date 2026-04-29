#!/bin/bash
make train dataset='aids' model='AidsPEGVAE'
make train dataset='mutagenicity' model='MutagenicityPEGVAE'
make train dataset='nci1' model='Nci1PEGVAE'
make train dataset='aids' model='AidsClassifier'
make train dataset='mutagenicity' model='MutagenicityClassifier'
make train dataset='nci1' model='Nci1Classifier'
