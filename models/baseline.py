import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):

    def __init__(self, output_size = 1024, image_channel_type = 'normi', use_embedding = True, trainable = False):
        super(ImageEncoder, self).__init__()

        self.image_channel_type = image_channel_type
        self.use_embedding      = use_embedding
        
        self.model              = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier   = nn.Sequential(*list(self.model.classifier)[:-1])
        
        self.fc    = nn.Sequential(
                         nn.Linear(4096, output_size),
                         nn.Tanh())
    
    def forward(self, images):
        if not self.use_embedding:
            images      = self.model(images)

        if self.image_channel_type == 'normi':
            images      = F.normalize(images, p = 2, dim = 1)
        image_embedding = self.fc(images)
        
        return image_embedding

class QuestionEncoder(nn.Module):
    
    def __init__(self, vocab_size = 10000, word_embedding_size = 300, hidden_size = 512, output_size = 1024,
                 num_layers = 2):
        super(QuestionEncoder, self).__init__()
        
        self.word_embeddings = nn.Sequential(
                                   nn.Embedding(vocab_size, word_embedding_size, padding_idx = 0),
                                   nn.Tanh())
        self.lstm            = nn.LSTM(input_size = word_embedding_size, hidden_size = hidden_size,
                                       num_layers = num_layers)
        # BIDIRECTIONAL??
        self.fc              = nn.Sequential(
                                   nn.Linear(2 * num_layers * hidden_size, output_size),
                                   nn.Tanh())
        
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
                 dropout_prob = 0.5, train_cnn = False):
        super(VQABaseline, self).__init__()
        
        self.word_embedding_size = word_embedding_size
        
        self.image_encoder       = ImageEncoder(output_size            = embedding_size,
                                                image_channel_type     = image_channel_type,
                                                use_embedding          = use_image_embedding,
                                                trainable              = train_cnn)
        self.question_encoder    = QuestionEncoder(vocab_size          = vocab_size,
                                                   word_embedding_size = word_embedding_size,
                                                   hidden_size         = lstm_hidden_size,
                                                   output_size         = embedding_size,
                                                   num_layers          = num_lstm_layers)
        
        self.mlp                 = nn.Sequential(
                                       nn.Linear(embedding_size, 1000),
                                       nn.Dropout(dropout_prob),
                                       nn.Tanh(),
                                       nn.Linear(1000, output_size))

    def forward(self, images, questions):
        image_embeddings    = self.image_encoder(images)
        question_embeddings = self.question_encoder(questions)
        final_embedding     = image_embeddings * question_embeddings
        
        output              = self.mlp(final_embedding)
        
        return output
