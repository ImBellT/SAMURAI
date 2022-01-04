import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn import preprocessing
from sklearn import model_selection

class Attacker:
    # CSVデータ処理（NumPy必要）
    def __init__(self, loading_file, data_normalize=True):
        csv_data = np.loadtxt(loading_file, delimiter=',', dtype=str)
        data_shape = csv_data.shape[0] - 1
        data = np.zeros((data_shape, 34, 2), dtype='int')
        meta = np.zeros((data_shape, 5), dtype='int')
        label = np.zeros(data_shape, dtype='U4')
        for i in range(data_shape):
            for index in range(34):
                data[i][index] = [int(csv_data[i + 1][index * 2 + 6]), int(csv_data[i + 1][index * 2 + 7])]
            meta[i] = csv_data[i + 1][:5].astype(int)
            label[i] = csv_data[i + 1][5]

        re_data = np.zeros((data.shape[0], 8))
        for num in range(data.shape[0]):
            # M-FC v1アーキテクチャ適用
            re_data[num][0] = encode_distance(encode_middle_shoulder(data[num]), data[num][9])  # ①左手と肩までの距離
            re_data[num][1] = encode_distance(encode_middle_shoulder(data[num]), data[num][10])  # ②右手と肩までの距離
            re_data[num][2] = encode_distance(encode_middle_mind(data[num]), data[num][9])  # ③左手と上半身下部までの距離
            re_data[num][3] = encode_distance(encode_middle_mind(data[num]), data[num][10])  # ④左手と上半身下部までの距離
            re_data[num][4] = encode_distance(encode_middle_shoulder(data[num]), data[num][15])  # ⑤左足と肩までの距離
            re_data[num][5] = encode_distance(encode_middle_shoulder(data[num]), data[num][16])  # ⑥右足と肩までの距離
            re_data[num][6] = encode_distance(encode_middle_mind(data[num]), data[num][15])  # ⑦左足と上半身下部までの距離
            re_data[num][7] = encode_distance(encode_middle_mind(data[num]), data[num][16])  # ⑧右足と上半身下部までの距離
        if data_normalize:
            sc = preprocessing.StandardScaler()
            re_data = sc.fit_transform(re_data)  # 学習データを正規化

        feature_names = ['左手と肩までの距離', '右手と肩までの距離', '左手と上半身下部までの距離', '左手と上半身下部までの距離', '左足と肩までの距離', '右足と肩までの距離', '左足と上半身下部までの距離', '右足と上半身下部までの距離']

        self.position_data = data
        self.data = re_data
        self.feature_names = feature_names
        self.label = label
        self.label_name_category = ['突き', '回し蹴り', '裏回し蹴り', '正蹴り', 'なし']
        self.label_name_position = ['上段', '中段']
        self.label_name_arrow = ['左', '右']
        self.label_name_status = ['有効', '無効']
        self.meta = meta

        print("学習データの読み込みに成功しました。")
        print("[読み込みデータ数：" + str(self.data.shape[0]) + "]")
        print("[特徴量数：" + str(self.data.shape[1]) + "]")


# 肩の中点を求める
def encode_middle_shoulder(part):
    x_pos = (part[22][0] + part[23][0]) / 2
    y_pos = (part[22][1] + part[23][1]) / 2
    return [x_pos, y_pos]


# 腹の点を求める
def encode_middle_mind(part):
    x_pos = (part[22][0] + part[23][0] + (part[28][0] + part[29][0]) * 2) / 6
    y_pos = (part[22][1] + part[23][1] + (part[28][1] + part[29][1]) * 2) / 6
    return [x_pos, y_pos]


# 2つの点の距離を求める（Numpy必要）
def encode_distance(a, b):
    dist_x = a[0] - b[0]
    dist_y = a[1] - b[1]
    distance = np.linalg.norm([dist_x, dist_y])
    return distance


attacker = Attacker('PASTE CSV FILE HERE')

x_train, y_train, x_test, y_test = model_selection.train_test_split(attacker.data, attacker.label, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(8))
model.add(tf.keras.layers.Dense(2048, activation='relu'))
model.add(tf.keras.layers.Dense(2048, activation='relu'))
model.add(tf.keras.layers.Dense(2048, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=1)

# モデルを保存
tfjs.converters.save_keras_model(model, "./tfjs_model")
