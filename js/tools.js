/**
 * 作成したモデルをダウンロードします
 * @param {object} model UseCustomModelで出力されたモデルデータ
 * @param {object} config 設定データ
 */
async function downloadModel(config, model) {

    // モデルからアーティファクトを作成（configはincludeOptimizerとtrainableOnlyのbool値を記載（なくてもいい））
    async function saveModelArtifacts(models, config) {
        // 重みデータを最適化
        const weightDataAndSpecs = await tf.io.encodeWeights(models[0].getNamedWeights(config));

        const returnString = false;
        const unusedArg = null;

        // モデル情報をJSON出力できるように変換する
        const modelConfig = models[0].toJSON(unusedArg, returnString);
        // アーティファクト定義
        const modelArtifacts = {
            modelTopology: modelConfig,
            format: 'layers-model',
            generatedBy: `Bell.T Science`,
            convertedBy: null,
        };
        // オプティマイザ追記
        const includeOptimizer = config == null ? false : config.includeOptimizer;
        if (includeOptimizer && models[0].optimizer != null) {
            modelArtifacts.trainingConfig = this.getTrainingConfig();
            const weightType = 'optimizer';
            const {
                data: optimizerWeightData,
                specs: optimizerWeightSpecs
            } = await tf.io.encodeWeights(await models[0].optimizer.getWeights(), weightType);
            weightDataAndSpecs.specs.push(...optimizerWeightSpecs);
            weightDataAndSpecs.data = tf.io.concatenateArrayBuffers([weightDataAndSpecs.data, optimizerWeightData]);
        }
        // メタデータ追記
        if (models[0].userDefinedMetadata != null) {
            const checkSize = true;
            checkUserDefinedMetadata(models[0].userDefinedMetadata, models[0].name, checkSize);
            modelArtifacts.userDefinedMetadata = models[0].userDefinedMetadata;
        }
        modelArtifacts.weightData = weightDataAndSpecs.data;
        modelArtifacts.weightSpecs = weightDataAndSpecs.specs;
        return modelArtifacts;
    }

    // アーティファクトに重み情報を追加
    function getModelForModelArtifacts(model_artifacts, w_manifest) {
        const data = {
            modelTopology: model_artifacts.modelTopology,
            format: model_artifacts.format,
            generatedBy: model_artifacts.generatedBy,
            convertedBy: model_artifacts.convertedBy,
            weightsManifest: w_manifest
        };
        if (model_artifacts.signature != null) {
            data.signature = model_artifacts.signature;
        }
        if (model_artifacts.userDefinedMetadata != null) {
            data.userDefinedMetadata = model_artifacts.userDefinedMetadata;
        }
        if (model_artifacts.modelInitializer != null) {
            data.modelInitializer = model_artifacts.modelInitializer;
        }
        if (model_artifacts.trainingConfig != null) {
            data.trainingConfig = model_artifacts.trainingConfig;
        }
        return data;
    }

    // アーティファクトからダウンロードデータを作成
    async function SaveModelTF(modelArtifact) {
        if (typeof (document) === 'undefined') {
            throw new Error('この環境上ではdocumentが存在しないため、ブラウザからのダウンロードはサポートされていません。');
        }

        async function modelBlob() {
            const modelJSON = getModelForModelArtifacts(modelArtifact, weightsManifest);
            return new Blob([JSON.stringify(modelJSON)], {type: 'application/json'});
        }

        async function weightBlob() {
            return new Blob([modelArtifact.weightData], {type: 'application/octet-stream'});

        }

        const weightsManifest = [{
            paths: ['./' + "samurai_custom_model_weights.bin"],
            weights: modelArtifact.weightSpecs
        }];
        return [await modelBlob(), await weightBlob()];
    }

    const transpose = a => a[0].map((_, c) => a.map(r => r[c])); // 転置配列を定義

    // JSZip定義
    const zip = new JSZip();

    async function generateData(model) {
        // 圧縮するファイルを定義
        const [, weights] = await SaveModelTF(await saveModelArtifacts(model));
        const std = new Blob([JSON.stringify(model[1], null, '')], {type: 'application\/json'});
        const data_train = new Blob([JSON.stringify(model[3][0], null, '')], {type: 'application\/json'});
        const data_test = new Blob([JSON.stringify(model[3][1], null, '')], {type: 'application\/json'});
        return [weights, std, data_train, data_test];
    }

    async function generateConfig(model) {
        const [model_data,] = await SaveModelTF(await saveModelArtifacts(model));
        const info_json = new Blob([JSON.stringify({
            "validationData": model[2].validationData,
            "params": model[2].params
        }, null, '')], {type: 'application\/json'});
        return [model_data, info_json];
    }

    function generateHistory(models) {

        let model_val_accuracy = [models[0][2].epoch];
        let model_val_loss = [], model_accuracy = [], model_loss = [];
        let model_val_accuracy_string = "", model_val_loss_string = "", model_accuracy_string = "",
            model_loss_string = "";
        for (let i = 0; i < models.length; i++) {
            model_val_accuracy_string += ",評価精度" + String(i + 1);
            model_val_accuracy = model_val_accuracy.concat([models[i][2].history.val_acc]);
            model_val_loss_string += ",評価損失" + String(i + 1);
            model_val_loss = model_val_loss.concat([models[i][2].history.val_loss]);
            model_accuracy_string += ",精度" + String(i + 1);
            model_accuracy = model_accuracy.concat([models[i][2].history.acc]);
            model_loss_string += ",損失" + String(i + 1);
            model_loss = model_loss.concat([models[i][2].history.loss]);
        }
        const model_history = transpose(model_val_accuracy.concat(model_val_loss, model_accuracy, model_loss));
        let csv_string = "学習回数" + model_val_accuracy_string + model_val_loss_string + model_accuracy_string + model_loss_string + "\r\n";
        model_history.forEach(d => {
            csv_string += d.join(","); // 配列ごとの区切りを「,」をつけて一列化
            csv_string += '\r\n';
        });
        let bom = new Uint8Array([0xEF, 0xBB, 0xBF]); // BOM付けてExcelの文字化けを防ぐ
        return new Blob([bom, csv_string], {type: "text/csv"}); // 抽出したデータをCSV形式に変換
    }

    async function generateCompCSV() {
        let model_result_all = [];
        for (let i = comp_model_range[1][0] - 1; i < comp_model_range[1][1]; i += comp_model_range[1][2]) {
            model_result_all[i] = await calcModelAverage(i);
        }
        let model_result_min = [], model_result_full = [];
        model_result_all.forEach(e => {
            const tmp = transpose(e);
            model_result_min = model_result_min.concat(tmp);
        });
        //共通項を書き込み
        for(let i=0; i < model_result_all[0][0].length; i++){
            model_result_full[i+1] = [];
            model_result_full[i+1][0] = model_result_all[0][1][i]; // 縦軸のユニット数を定める
        }
        model_result_full[0] = []; // 見出し用に初期化
        for(let i=0; i < model_result_all.length; i++){
            model_result_full[0][i+1] = model_result_all[i][0][0]; // 横軸の層数を定める
        }
        const label = ["評価精度", "評価損失", "精度", "損失"];
        let model_result_full_all = [[], [], [], []];
        for (let x = 0; x < label.length; x++) { // 評価項目ごとに実行
            model_result_full[0][0] = label[x];
            // 測定データを書き込み
            for(let i=0; i < model_result_all[0][0].length; i++){
                for(let j=0; j < model_result_all.length; j++){
                    model_result_full[i+1][j+1] = model_result_all[j][2+x][i]; // i+1が縦列（ユニット数で見出しを飛ばしている）、jが横列でレイヤー数に準ずる。そこでallの配列に沿う。
                }
            }
            model_result_full_all[x] = JSON.parse(JSON.stringify(model_result_full));
        }

        let csv_string_min = "層数,ユニット数,評価精度,評価損失,精度,損失\r\n";
        model_result_min.forEach(d => {
            csv_string_min += d.join(","); // 配列ごとの区切りを「,」をつけて一列化
            csv_string_min += '\r\n';
        });
        let csv_string_full = [];
        for(let i=0; i < model_result_full_all.length; i++){
            csv_string_full[i] = "";
            model_result_full_all[i].forEach(d => {
                csv_string_full[i] += d.join(","); // 配列ごとの区切りを「,」をつけて一列化
                csv_string_full[i] += '\r\n';
            });
        }
        let bom = new Uint8Array([0xEF, 0xBB, 0xBF]); // BOM付けてExcelの文字化けを防ぐ
        const min = new Blob([bom, csv_string_min], {type: "text/csv"}); // 抽出したデータをCSV形式に変換
        const full_val_acc = new Blob([bom, csv_string_full[0]], {type: "text/csv"});
        const full_val_loss = new Blob([bom, csv_string_full[1]], {type: "text/csv"});
        const full_acc = new Blob([bom, csv_string_full[2]], {type: "text/csv"});
        const full_loss = new Blob([bom, csv_string_full[3]], {type: "text/csv"});
        return [min, full_val_acc, full_val_loss, full_acc, full_loss];
    }

    if (config.comp_model) {
        // モデル比較用
        const output = await generateCompCSV();
        zip.file("データ概要.csv", output[0]);
        zip.file("評価精度.csv", output[1]);
        zip.file("評価損失.csv", output[2]);
        zip.file("精度.csv", output[3]);
        zip.file("損失.csv", output[4]);
    } else {
        // 定義したファイルをアーカイブに追加
        let part = zip.folder("モデルデータ・学習データ");
        for (let i = 0; i < model.length; i++) {
            const [data_train, data_test] = await generateData(model);
            let x = part.folder(String(i));
            x.file("学習データ.json", data_train);
            x.file("テストデータ.json", data_test);
        }
        const [model_data, info_json] = await generateConfig(model);
        zip.file("model.json", model_data);
        zip.file("モデル仕様.json", info_json);
        zip.file("学習精度・損失.csv", generateHistory(model));
    }
    zip.generateAsync({type: "blob"}).then(function (dataBlob) {
        const DownloadUrl = URL.createObjectURL(dataBlob); // BlobデータをURLに変換
        const downloadOpen = document.createElement('a');
        downloadOpen.href = DownloadUrl;
        downloadOpen.download = "独自学習モデル出力 - SAMURAI"; // ダウンロード時のファイル名を指定
        downloadOpen.click(); // 疑似クリック
        URL.revokeObjectURL(DownloadUrl); // 作成したURLを解放（削除）
    });
}

/**
 * 独自で作成したモデルを読み込みます
 * @return {object} 作成したモデル
 */
function MakeCustomModel() {
    let function_str = editor_a.getValue();
    function_str = "function custom_model(){\n" + function_str + "\n}";
    const custom_model_tag = document.createElement('script');
    custom_model_tag.innerHTML = function_str;
    document.body.appendChild(custom_model_tag);
    // noinspection JSUnresolvedFunction
    return custom_model();
}

/**
 * デバッグ用のモデル定義
 */
function compModel(layer_num, unit_num) {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: unit_num, activation: 'relu', inputShape: [16]}));
    for (let i = 0; i < layer_num; i++) model.add(tf.layers.dense({units: unit_num, activation: 'relu'}));
    model.add(tf.layers.dense({units: 4, activation: 'softmax'}));
    model.compile({
        optimizer: tf.train.adam(0.001, 0.9, 0.999),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

/**
 * 独自モデルを学習します
 * @param {string} csv_file CSVファイル形式の学習データ（URLでファイルを指定・Input形式の場合createObjectURLで作る）
 * @param {string} csv_file_test CSVファイル形式の評価用テストデータ（URLでファイルを指定・Input形式の場合createObjectURLで作る）
 * @param {object} num 学習の回数を指定します（HTML表記にのみ影響）
 * @param {object} config 設定データ
 * @param layers 層数（モデル比較時のみ）
 * @param units ユニット数（モデル比較時のみ）
 * @return {object} データによって学習済みのモデル
 */
async function learnModel(csv_file, csv_file_test, num, config, layers = 4, units = 512) {
    _statusText.main.innerHTML = "処理状況：新規モデルの作成・学習を実行します。";
    const model = config.comp_model ? compModel(layers, units) : MakeCustomModel();
    let param = {
        mean: [],
        std: [],
        res: [
            Number(document.getElementById("pointing_res_w").value),
            Number(document.getElementById("pointing_res_h").value)
        ],
        selector: [
            document.getElementById("sel-1").checked ? 1 : 0,
            document.getElementById("sel-2").checked ? 1 : 0,
            document.getElementById("sel-3").checked ? 1 : 0,
            document.getElementById("sel-4").checked ? 1 : 0
        ],
        FC_model_version: 1
    }

    const [data, data_test] = config.custom_valid ? [await ProcessCSVData(csv_file), await ProcessCSVData(csv_file_test)] : splitData(await ProcessCSVData(csv_file), config, param.selector);

    const [x_pos, y] = SplitDataIndex(data);
    let x_key = FeatureConvertAll(x_pos, param);
    const [x_pos_test, y_test] = SplitDataIndex(data_test);
    let x_key_test = FeatureConvertAll(x_pos_test, param);

    [param.mean, param.std] = CalcMeanAndStd(x_key); // 平均値や標準偏差を導出

    // デバッグ用
    if (config.debug_mode) {
        console.log("モデルサマリー↓");
        model.summary();
        console.log("学習データ↓");
        console.log(data);
        console.log("テストデータ↓");
        console.log(data_test);
        console.log("説明変数↓");
        console.log(x_key);
        console.log("平均値↓");
        console.log(JSON.parse(JSON.stringify(param.mean)));
        console.log("標準偏差↓");
        console.log(JSON.parse(JSON.stringify(param.std)));
    }

    x_key = KeyStandardization(x_key, param);
    x_key_test = KeyStandardization(x_key_test, param);
    const x_tensor = tf.tensor2d(KeyToValue(x_key));
    const y_tensor = tf.tensor2d(LabelOneHotEncode(y, param));

    function onEpochEnd(epoch, logs) {
        switch (num[0]) {
            case -1: // シャッフルなし
                _statusText.main.innerHTML = "処理状況：新規モデルを学習中…<br/>・Epoch " + epoch + "<br/>・学習データ精度（Accuracy）" + logs.acc + "<br/>・学習データ損失（Loss）" + logs.loss + "<br/>・テストデータ精度（Val Accuracy）" + logs.val_acc + "<br/>・テストデータ損失（Val Loss）" + logs.val_loss;
                break;
            case 0: // シャッフルのみ
                _statusText.main.innerHTML = "処理状況：新規モデルを学習中…<br/>・シード数：" + config.seed + "（" + (num[1] + 1) + "回目）<br/>・Epoch " + epoch + "<br/>・学習データ精度（Accuracy）" + logs.acc + "<br/>・学習データ損失（Loss）" + logs.loss + "<br/>・テストデータ精度（Val Accuracy）" + logs.val_acc + "<br/>・テストデータ損失（Val Loss）" + logs.val_loss;
                break;
            case 1: // シャッフル＆モデル比較
                _statusText.main.innerHTML = "処理状況：新規モデルを学習中…<br/>・層数：" + (num[1][0] + 1) + "　ユニット数：" + num[1][1] + "<br/>・シード数：" + config.seed + "（" + (num[1][2] + 1) + "回目）<br/>・Epoch " + epoch + "<br/>・学習データ精度（Accuracy）" + logs.acc + "<br/>・学習データ損失（Loss）" + logs.loss + "<br/>・テストデータ精度（Val Accuracy）" + logs.val_acc + "<br/>・テストデータ損失（Val Loss）" + logs.val_loss;
                break;
            default:
                _statusText.main.innerHTML = "学習時にエラーが発生しました。（Error：学習カテゴリの区分不適合）";
        }
        if (document.getElementById("debug-mode").checked) {
            console.log(String(epoch) + '回目の結果↓');
            console.log(logs);
        }
    }

    const info = await model.fit(x_tensor, y_tensor, {
        batchSize: Number(document.getElementById("batch_size").value),
        epochs: Number(document.getElementById("epoch_num").value),
        validationData: [tf.tensor2d(KeyToValue(x_key_test)), tf.tensor2d(LabelOneHotEncode(y_test, param))],
        callbacks: {onEpochEnd}
    });

    // デバッグ用
    if (config.debug_mode) {
        console.log("学習結果↓");
        console.log(JSON.parse(JSON.stringify(info)));
    }

    return [model, param, info, [data, data_test]];
}

/**
 * 独自モデルを準備します
 * @param {string} csv_file CSVファイル形式の学習データ（URLでファイルを指定・Input形式の場合createObjectURLで作る）
 * @param {string} csv_file_test CSVファイル形式の評価用テストデータ（URLでファイルを指定・Input形式の場合createObjectURLで作る）
 * @param {object} config 設定データ
 * @return {object} データによって学習済みのモデル
 */
async function UseCustomModel(csv_file, csv_file_test, config) {

    if (isNaN(config.seed)) config.seed = Math.floor(Math.random() * 100);

    if (!config.data_shuffle && !config.data_shuffle_comp) return await learnModel(csv_file, csv_file_test, [-1], config);

    const [learning_mode, layer_num_range, unit_num_range] = (() => {
        if (config.comp_model) {
            return comp_model_range; // シャッフル＆モデル比較（始点はどっちも1以上が必要）
        } else if (!config.custom_valid && config.data_shuffle && config.data_shuffle_comp) {
            return [0, [1, 1, 1], [1, 1, 1]]; // シャッフルのみ
        } else {
            throw new Error("学習方法の設定時にエラーが発生しました。");
        }
    })();
    window.indexedDB.deleteDatabase("samurai_DB_model");
    let openReq = indexedDB.open("samurai_DB_model", 1);
    openReq.onupgradeneeded = function (event) {
        let db = event.target.result; // データベースを定義
        for (let i = layer_num_range[0]; i <= layer_num_range[1]; i += layer_num_range[2]) {
            db.createObjectStore("layer" + ('0000' + i).slice(-4), {keyPath: "key"});　// 作成したモデルを格納
        }
    }

    const seed_origin = config.seed;

    for (let layer_num = layer_num_range[0] - 1; layer_num < layer_num_range[1]; layer_num += layer_num_range[2]) { // 層数
        for (let unit_num = unit_num_range[0]; unit_num <= unit_num_range[1]; unit_num += unit_num_range[2]) { // ユニット数
            let param_tmp = [], info_tmp = [], data_tmp = [];
            config.seed = seed_origin;
            for (let seed_num = 0; seed_num < config.shuffle_comp_times; seed_num++) { // シード数
                [, param_tmp[seed_num], info_tmp[seed_num], data_tmp[seed_num]] = await learnModel(csv_file, csv_file_test, [learning_mode, [layer_num, unit_num, seed_num]], config, layer_num, unit_num);
                config.seed++;
            }
            await insertPoseDB({
                param: param_tmp,
                info: info_tmp,
                data: data_tmp
            }, "samurai_DB_model", "layer" + ('0000' + (layer_num + 1)).slice(-4), unit_num); // ユニットごとに切り分ける
        }
    }

    _statusText.main.innerHTML = "処理状況：新規モデルの学習が完了しました。ならびにモデルのダウンロードを解除しました。";
    return 0;
}

/**
 * 乱数シードによる学習から損失・精度の平均を計算します
 * @return {object} モデルの平均値・シード数
 */
async function calcModelAverage(layer_num) {
    const resume_data = await resumePoseDB("samurai_DB_model", "layer" + ('0000' + (layer_num + 1)).slice(-4));

    let layers_num = [], units_num = [], model_val_accuracy = [], model_val_loss = [], model_accuracy = [],
        model_loss = [];
    const epoch_length = resume_data[0].data.info.length;
    for (let i = 0; i < resume_data.length; i++) {
        units_num[i] = resume_data[i].key;
        layers_num[i] = layer_num + 1;
        model_val_accuracy[i] = 0, model_val_loss[i] = 0, model_accuracy[i] = 0, model_loss[i] = 0;
        resume_data[i].data.info.forEach(e => {
            model_val_accuracy[i] += e.history.val_acc.slice(-1)[0] / epoch_length; // 評価精度の最終結果のみを加算する
            model_val_loss[i] += e.history.val_loss.slice(-1)[0] / epoch_length;
            model_accuracy[i] += e.history.acc.slice(-1)[0] / epoch_length;
            model_loss[i] += e.history.loss.slice(-1)[0] / epoch_length;
        });
    }
    return [layers_num, units_num, model_val_accuracy, model_val_loss, model_accuracy, model_loss]; // 各結果の平均値（[ユニットごとの評価精度平均（指定シード数から導出）]）
}

/**
 * MIYABIモデルを読み込みます（binファイルとstandard.jsonファイルが同一ディレクトリにいることを必ず確認！）
 * @param {string} category モデルの種類
 * @return {object} [Kerasモデル, モデルのパラメータ]
 */
async function miyabiSelector(category) {
    const version = Number(category.substr(8, 1));
    const optimizer = category.substr(10);
    console.log("/model/miyabi_v" + version + "/" + optimizer + "/model.json");
    return LoadModel("/model/miyabi_v" + version + "/" + optimizer + "/model.json");
}

/**
 * 学習モデルを読み込みます（binファイルとstandard.jsonファイルが同一ディレクトリにいることを必ず確認！）
 * @param {object} model_file model.jsonのパス（外部も可）
 * @return {object} [Kerasモデル, モデルのパラメータ]
 */
async function LoadModel(model_file) {
    const dnn_model = await tf.loadLayersModel(model_file);
    const std_file = model_file.split("/").reverse().slice(1).reverse().join("/") + "/standard.json";
    const param_str = await GetDataFromFile(std_file);
    const param = JSON.parse(param_str);
    return [dnn_model, param];
}

/**
 * 学習モデルをjson, bin, standard別個で読み込みます
 * @param {object} json モデルjsonファイルのパス
 * @param {object} weight binファイルのパス
 * @param {object} standard standardファイルのパス
 * @return {object} [Kerasモデル, モデルのパラメータ]
 */
async function LoadModel_Outside(json, weight, standard) {
    const dnn_model = await tf.loadLayersModel(tf.io.browserFiles([json.files[0], weight.files[0]]));
    const DownloadUrl = URL.createObjectURL(standard.files[0]);
    const param_str = await GetDataFromFile(DownloadUrl);
    const param = await JSON.parse(param_str);
    return [dnn_model, param];
}

/**
 * 読み込んだテキストファイル（CSV・JSON）を改行（\n）含めた文字列にして出力します
 * @param {string} URL CSV・JSONファイルのパス（外部でも可）
 * @return {promise} 変換後の文字列
 */
async function GetDataFromFile(URL) {
    return new Promise(resolve => {
        let req = new XMLHttpRequest(); // HTTPでファイルを読み込むためのXMLHttpRrequestオブジェクトを生成
        req.open("get", URL, true); // アクセスするファイルを指定
        req.send(null); // HTTPリクエストの発行
        req.onload = () => {
            resolve(req.responseText);
        }
    });
}

/**
 * 読み込んだテキストファイル（CSV）を（position（x, y, score）×2）（label）（meta）の3オブジェクトにして出力します
 * @param {string} URL CSVファイルのパス（外部でも可）
 * @return {object} 変換後のデータ配列
 */
async function ProcessCSVData(URL) {
    let csv_data = await GetDataFromFile(URL);
    // 文字列をデータ化
    let result = [];
    let tmp = csv_data.split('\n'); // 改行を区切り文字として行を要素とした配列を生成
    for (let i = 0; i < tmp.length; ++i) {
        result[i] = tmp[i].split(','); // 各行ごとにカンマで区切った文字列を要素とした二次元配列を生成
    }

    // データを振り分け
    let data = [];
    for (let i = 0; i < result.length - 2; i++) {
        data[i] = {position: [[], []], meta: [], label: ""}; // 2人分の骨格座標、メタ情報配列、目標ラベル文字列を確保
        for (let j = 0; j < 17; j++) {
            data[i].position[0][j] = {};
            data[i].position[1][j] = {};
            data[i].position[0][j].x = Number(result[i + 1][j * 2 + 6]);
            data[i].position[0][j].y = Number(result[i + 1][j * 2 + 7]);
            data[i].position[1][j].x = Number(result[i + 1][j * 2 + 40]);
            data[i].position[1][j].y = Number(result[i + 1][j * 2 + 41]);
            switch (result[i + 1][j + 74]) {
                case 'CONFIRM':
                    data[i].position[0][j].score = 1.0;
                    break;
                case 'UNCLEAR':
                    data[i].position[0][j].score = 0.7;
                    break;
                case 'OUT_OF_RANGE':
                    data[i].position[0][j].score = 0.5;
            }
            switch (result[i + 1][j + 91]) {
                case 'CONFIRM':
                    data[i].position[1][j].score = 1.0;
                    break;
                case 'UNCLEAR':
                    data[i].position[1][j].score = 0.7;
                    break;
                case 'OUT_OF_RANGE':
                    data[i].position[1][j].score = 0.5;
            }
        }
        for (let j = 0; j < 5; j++) {
            data[i].meta[j] = Number(result[i + 1][j]);
        }
        data[i].label = result[i + 1][5];
    }

    return data;

}

/**
 * 学習データをtrainとtestで分割します（目標セレクタごと）
 * @param {object} data シャッフルしたい配列
 * @param {object} config 設定データ
 * @param {object} selector 目標セレクタ
 * @return {object} [学習データ, テストデータ]
 */
function splitData(data, config, selector) {
    let selector_str = [[1, 2, 3, 4], ['U', 'M'], ['L', 'R'], ['S', 'F']];
    let tmp, result = [[], []];
    if (config.data_shuffle) data = shuffleMatrix(data, config.seed); // シャッフルの指定があれば事前に行う
    if (selector.toString() === [1, 0, 0, 0].toString()) { // toStringを使う理由はJavaScriptでは配列の合同関係を直接検証することができないため（これがないと常にfalseを返してしまう）
        tmp = [[], [], [], []];
        data.forEach(e => {
            const cat = selector_str[0].indexOf(Number(e.label.charAt(0)));
            tmp[cat] = tmp[cat].concat(e)
        }); // 技種別コードの配列に突っ込む
    } else {
        // 後で頑張れ for 未来の自分
    }
    tmp.forEach(e => {
        result[0] = result[0].concat(e.slice(Math.ceil(e.length * config.test_ratio), e.length));
        result[1] = result[1].concat(e.slice(0, Math.ceil(e.length * config.test_ratio)));
    });
    return result;
}

/**
 * 乱数を発生させ、配列をシャッフルします（シード機能あり）
 * @param {number} seed 乱数シード値（NaN時は完全ランダム）
 * @param {object} matrix シャッフルしたい配列
 * @return {object} シャッフル後の配列
 */
function shuffleMatrix(matrix, seed) {

    seed *= 135485; // 適当にシード値を増やしておく

    // 乱数シードリストを定義
    let seed_list = [123456789, 362436069, 521288629, seed];
    let new_list_index = [];
    let new_matrix = [];

    function make() {
        const t = seed_list[0] ^ (seed_list[0] << 11);
        seed_list[0] = seed_list[1];
        seed_list[1] = seed_list[2];
        seed_list[2] = seed_list[3];
        seed_list[3] = (seed_list[3] ^ (seed_list[3] >>> 19)) ^ (t ^ (t >>> 8));
        const r = Math.abs(seed_list[3]);
        return r % (matrix.length + 1);
    }

    for (let i = 0; i < matrix.length; i++) {
        while (true) {
            const num = make();
            let usable = false;
            new_list_index.forEach(e => {
                if (e === num) usable = true
            });
            if (!usable) {
                new_list_index = new_list_index.concat([[i, num]]);
                break;
            }
        }
    }
    new_list_index.sort((a, b) => a[1] - b[1]);
    matrix.forEach(e => new_matrix[matrix.indexOf(e)] = matrix[new_list_index[matrix.indexOf(e)][0]]);
    return new_matrix;
}

/**
 * CSVでインポートしたデータ配列を座標、目標ラベル、メタデータで切り分けます
 * @param {object} data 振り分け前のデータ配列（GetDataFromCSVから得たデータ形式が必要）
 * @param {boolean} use_position_score 切り分け時に精度を保持するか
 * @return {object} 振り分け後のデータ（position(x, y (,score) ×2), label, meta）
 */
function SplitDataIndex(data, use_position_score = true) {
    let position = [];
    let label = [];
    let meta = [];
    for (let i = 0; i < data.length; i++) {
        position[i] = [[], []];
        for (let j = 0; j < data[i].position[0].length; j++) {
            position[i][0][j] = {};
            position[i][1][j] = {};
            position[i][0][j].x = data[i].position[0][j].x;
            position[i][0][j].y = data[i].position[0][j].y;
            position[i][1][j].x = data[i].position[1][j].x;
            position[i][1][j].y = data[i].position[1][j].y;
            if (use_position_score) {
                position[i][0][j].score = data[i].position[0][j].score;
                position[i][1][j].score = data[i].position[1][j].score;
            }
        }
        for (let j = 0; j < data[i].meta.length; j++) {
            meta[i] = data[i].meta;
        }
        label[i] = data[i].label;
    }
    return [position, label, meta];
}