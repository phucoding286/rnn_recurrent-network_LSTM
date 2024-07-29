import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences

# lớp chú ý bahdanau
class Attention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        """
        ở đây trọng số chú ý không được chứa bias.
        trọng số W1 giúp học đầu ra decoder của mạng rnn.
        trọng số W2 học đầu ra của encoder của rnn.
        sau đó được đi qua hàm tanh để tính điểm chú ý.
        sau đó điểm chú ý sẽ được trọng số V học và đi qua hàm softmax tạo ra một điểm
        chú ý (attention scores).
        """
        self.W1 = layers.Dense(units=units, use_bias=False)
        self.W2 = layers.Dense(units=units, use_bias=False)
        self.V = layers.Dense(units=units, use_bias=False)
    
    def call(self, q, k):
        # ma trận Q (đầu ra decoder rnn) được trọng số W1 học
        # ma trận K (đầu ra encoder rnn) được trọng số W2 học
        W1 = self.W1(q)
        W2 = self.W2(k)

        # thêm một chiều để khi cuối reduce sum ma trận values đầu ra không bị cắt bỏ chiều quan trọng
        W2 = tf.expand_dims(W2, 1)

        # tính điểm chú ý đi qua hàm tanh và được trọng số V học sau đó đi qua softmax để 
        # hoàn thành tạo điểm chú ý
        scores = tf.nn.tanh(W1 + W2)
        scores = self.V(scores)
        weights = tf.nn.softmax(scores, axis=-1)

        """
        nhân điểm chú ý với ma trận q (đầu ra decoder rnn) để lấy giá trị cuối cùng (attention values)
        attention values dùng để concat (kết hợp) với ma trận encoder rnn đầu ra sau đó đi ra với lớp
        tuyến tính để tạo ra kết quả dự đoán
        """
        values = weights * q
        values = tf.reduce_sum(values, axis=1)

        return values

# lớp LSTM kết hợp Attention
class AttentionLSTM(keras.Model):
    def __init__(self, vocab_size, num_enc_layer=1, num_dec_layer=1, units=128):
        super().__init__()
        # lớp nhúng để lấy vector đại diện cho vị trí từ (index word)
        self.embedding = keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=512,
                                                trainable=True)
        
        # các lớp rnn encoder được tạo nhiều hơn bởi vòng lặp for
        self.lstm_encs = [keras.layers.LSTM(activation="tanh",
                                  recurrent_activation="sigmoid",
                                  units=units,
                                  dropout=0.001,
                                  recurrent_dropout=0.001,
                                  return_sequences=True,
                                  return_state=True
                                  ) for _ in range(num_enc_layer)]
        
        # các lớp rnn decoder được tạo nhiều hơn bởi vòng lặp for
        self.lstm_decs = [keras.layers.LSTM(activation="tanh",
                                  recurrent_activation="sigmoid",
                                  units=units,
                                  dropout=0.001,
                                  recurrent_dropout=0.001,
                                  return_sequences=True,
                                  return_state=True
                                  ) for _ in range(num_dec_layer)]
        
        # lớp attention để tính giá trị chú ý
        self.attention = Attention(units)
        
        # lớp tuyến tính đầu ra với hàm kích hoạt softmax để tạo ra phân phối xác suất dự đoán
        # cho từng từ
        self.dense_out = keras.layers.Dense(units=vocab_size,
                                            activation="softmax",
                                            use_bias=True)

    # hàm call cho lớp
    def call(self, x):
        # truyền x (dữ liệu) qua lớp embedding để lấy biểu diễn nhúng
        x = self.embedding(x)
        
        # chuyền dữ liệu biểu diễn nhúng thẳng qua lớp rnn encoder
        for lstm_enc in self.lstm_encs:
            x_out_enc, hid, cell = lstm_enc(x)

        # chuyền dữ liệu biểu diễn mã hóa (đầu ra encoder rnn) qua decoder rnn
        for lstm_dec in self.lstm_decs:
            x_out_dec, hid, cell = lstm_dec(x, initial_state=(hid, cell))
        
        # chuyền đầu ra encoder rnn và đầu ra decoder rnn vào lớp chú ý để lấy giá trị chú ý
        val = self.attention(x_out_dec, x_out_enc)
        # kết hợp giá trị chú ý với đầu ra encoder rnn
        val = tf.concat([x_out_dec, val], -1)
        
        # chuyền dữ liệu đã đi qua từng lớp trên qua lớp tuyến tính có hàm kích hoạt softmax
        # để tạo ra ma trận phân phối xác suất cho từng từ
        out = self.dense_out(val)
        return out

# lớp tokennizer + lớp attention và lớp lstm
class TokenizerAttentionLSTM():
    def __init__(self, num_enc_layer, num_dec_layer, units,
                 char_level=False,
                 num_word=1000,
                 filters='!"#$%&()*+./;<=>@[\\]^_`{|}~'):
        
        """
        tham số char_level với định dạng bool: True/False dùng để tách từng từ làm index
        hoặc không (mặc định tách từng từ).

        tham số num_word dùng chỉ giới hạn tối đa số lượng tokenizer mà lớp tokenizer của tensorflow
        có thể chứa.

        tham số filters dùng để loại bỏ ký tự đặc biệt
        """
        self.tokenizer = Tokenizer(char_level=char_level, num_words=num_word, 
                              filters=filters
                )
        self.x_enc, self.y_enc = self.tokenized()
        self.rnn_model = AttentionLSTM(len(self.tokenizer.word_index)+1,
                                  num_enc_layer=num_enc_layer,
                                  num_dec_layer=num_dec_layer,
                                  units=units
                                )
    
    # mở file lấy từng câu và tiến hành áp dụng tokenizer lên
    def tokenized(self, file_x='x.txt', file_y="y.txt"):

        # mở file và lấy dữ liệu
        with open(file_x, "r", encoding="utf8") as f:
            x = f.read().splitlines()
        with open(file_y, "r", encoding="utf8") as f:
            y = f.read().splitlines()
        
        # fit tokenizer với tổng dữ liệu để tokenizer có được đủ thông tin để hoạt động
        self.tokenizer.fit_on_texts(x + y)
        # bắt đầu mã hóa những lô văn bản x (chuổi nguồn) y (đích) thành biểu diễn index (số)
        x_enc = self.tokenizer.texts_to_sequences(x)
        y_enc = self.tokenizer.texts_to_sequences(y)

        # tiến hành thêm đệm để đảm bảo ma trận cùng kích thước, với tham số maxlen
        # là chiều dài tối đa của chiều đặc trưng
        x_enc_pad = pad_sequences(x_enc, maxlen=50, padding="post", truncating="post")
        y_enc_pad = pad_sequences(y_enc, maxlen=50, padding="post", truncating="post")

        return x_enc_pad, y_enc_pad
    
    # hàm training mô hình
    def training(self):
        """
        biên dịch mô hình với hàm loss là sparse_categorical_crossentropy
        hàm loss này có nghĩa là dùng cho phân loại đa lớp.
        hàm tối ưu hóa là adam.
        tham số metrics dùng để hiển thông số độ chính xác trong quá trình huấn luyện.
        """
        self.rnn_model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["acc"]
                        )
        # training mô hình với hàm fit
        self.rnn_model.fit(self.x_enc, self.y_enc, batch_size=16, epochs=100)
    
    # hàm dự đoán từ lô riêng lẻ
    def predict(self, x):
        x_test = self.tokenizer.texts_to_sequences([f"input: {x}"])
        x_test = pad_sequences(x_test, maxlen=50, padding="post", truncating="post")

        model_response = self.rnn_model(x_test)
        tensor_response = tf.argmax(model_response, -1).numpy()
        text_response = self.tokenizer.sequences_to_texts(tensor_response)

        return text_response, tensor_response, model_response

model = TokenizerAttentionLSTM(num_enc_layer=1,
                               num_dec_layer=1,
                               units=512,
                               char_level=False,
                               num_word=10000
                            )
model.training()

his_inp = ""
his_output = ""

while True:
    you = input("You: ")
    inp = f"his inp: {his_inp} - his output: {his_output} - input: {you}"

    output = model.predict(inp)
    print(output[0])

    his_inp = you
    his_output = output[0]
