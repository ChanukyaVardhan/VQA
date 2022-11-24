import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):

    def __init__(self, output_size = 1024, image_channel_type = 'normi', use_embedding = True, trainable = False,
                 dropout_prob = 0.5, use_dropout = False):
        super(ImageEncoder, self).__init__()

        self.image_channel_type = image_channel_type
        self.use_embedding      = use_embedding
        
        self.model              = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier   = nn.Sequential(*list(self.model.classifier)[:-1])
        
        self.fc    = nn.Sequential()
        self.fc.append(nn.Linear(4096, output_size))
        if use_dropout:
            self.fc.append(nn.Dropout(dropout_prob))
        self.fc.append(nn.Tanh())
    
    def forward(self, images):
        if not self.use_embedding:
            images      = self.model(images)

        if self.image_channel_type == 'normi':
            images      = F.normalize(images, p = 2, dim = 1)
        image_embedding = self.fc(images)
        
        return image_embedding

class QuestionEncoder(nn.Module):
    
    def __init__(self, vocab_size = 10000, word_embedding_size = 300, hidden_size = 512, output_size = 1024,
                 num_layers = 2, dropout_prob = 0.5, use_dropout = False):
        super(QuestionEncoder, self).__init__()
        
        self.word_embeddings = nn.Sequential()
        self.word_embeddings.append(nn.Embedding(vocab_size, word_embedding_size, padding_idx = 0))
        if use_dropout:
            self.word_embeddings.append(nn.Dropout(dropout_prob))
        self.word_embeddings.append(nn.Tanh())

        self.lstm            = nn.LSTM(input_size = word_embedding_size, hidden_size = hidden_size,
                                       num_layers = num_layers)

        self.fc              = nn.Sequential()
        self.fc.append(nn.Linear(2 * num_layers * hidden_size, output_size))
        if use_dropout:
            self.fc.append(nn.Dropout(dropout_prob))
        self.fc.append(nn.Tanh())
        
    def forward(self, questions):
        x                  = self.word_embeddings(questions)
        x                  = x.transpose(0, 1)
        _, (hidden, cell)  = self.lstm(x)
        x                  = torch.cat((hidden, cell), 2)
        x                  = x.transpose(0, 1)
        x                  = x.reshape(x.size()[0], -1)
        x                  = nn.Tanh()(x)
        question_embedding = self.fc(x)
        
        return question_embedding

class VQABaseline(nn.Module):

    def __init__(self, vocab_size = 10000, word_embedding_size = 300, embedding_size = 1024, output_size = 1000,
                 lstm_hidden_size = 512, num_lstm_layers = 2, image_channel_type = 'normi', use_image_embedding = False,
                 dropout_prob = 0.5, train_cnn = False, use_dropout = False):
        super(VQABaseline, self).__init__()
        
        self.word_embedding_size = word_embedding_size
        
        self.image_encoder       = ImageEncoder(output_size            = embedding_size,
                                                image_channel_type     = image_channel_type,
                                                use_embedding          = use_image_embedding,
                                                trainable              = train_cnn,
                                                dropout_prob           = dropout_prob,
                                                use_dropout            = use_dropout)
        self.question_encoder    = QuestionEncoder(vocab_size          = vocab_size,
                                                   word_embedding_size = word_embedding_size,
                                                   hidden_size         = lstm_hidden_size,
                                                   output_size         = embedding_size,
                                                   num_layers          = num_lstm_layers,
                                                   dropout_prob        = dropout_prob,
                                                   use_dropout         = use_dropout)
        
        self.mlp                 = nn.Sequential()
        self.mlp.append(nn.Linear(embedding_size, 1000))
        if use_dropout:
            self.mlp.append(nn.Dropout(dropout_prob))
        self.mlp.append(nn.Tanh())
        self.mlp.append(nn.Dropout(dropout_prob)) # part of the base line model by default
        self.mlp.append(nn.Linear(1000, output_size))

    def forward(self, images, questions):
        image_embeddings    = self.image_encoder(images)
        question_embeddings = self.question_encoder(questions)
        final_embedding     = image_embeddings * question_embeddings
        
        output              = self.mlp(final_embedding)
        
        return output
