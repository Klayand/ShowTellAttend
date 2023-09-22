import torch
import torch.nn as nn
import statistics
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.encoder = models.inception_v3(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features, _ = self.encoder(images)
        features = self.relu(features)
        features = self.dropout(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # [word_length, batch_size, embed_size]
        embeddings = self.dropout(embeddings)  # features [batch_size, embed_size]
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)  # [word_length, batch_size, embed_size]
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, cfg):
        super(CNNtoRNN, self).__init__()
        self.embed_size = cfg.EMBED_SIZE
        self.hidden_size = cfg.HIDDEN_SIZE
        self.vocab_size = cfg.VOCAB_SIZE
        self.num_layers = cfg.NUM_LAYERS

        self.encoderCNN = EncoderCNN(self.embed_size)
        self.decoderRNN = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))

                predicted = output.argmax(1)
                result_caption.append(predicted.item())

                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
