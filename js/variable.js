/* 変数を定義 */
// 主要変数
let vid_file = document.getElementById("vid_file"); // inputファイル
let _inputVideo = document.getElementById("input_vid"); // 処理用videoタグ
let position_data; // 骨格データを保存する
let result_data; // 骨格データを保存する
let model_data = []; // 自作モデルを保存する

// キャンバス
let _Canvas = {
    calc: document.getElementById("calc_vid"),
    preview: document.getElementById("output_vid")
}
let _Ctx = {
    calc: _Canvas.calc.getContext("2d"),
    preview: _Canvas.preview.getContext("2d")
};

// ステータス文
let _statusText = {
    main: document.getElementById("status"),
    origin: document.getElementById("origin_param"),
    master: document.getElementById("master_param")
};

// 骨格推定変数
let _poseParam = {
    model: document.getElementById("pose_model"),
    stride: document.getElementById("output_stride"),
    quant: document.getElementById("quant_bytes"),
    input_res: document.getElementById("input_res"),
    flame_stride: document.getElementById("flame_stride"),
    threshold: document.getElementById("input_threshold")
};

// スタイル関連
let _Switch = [false, false];
let editor_a = ace.edit("editor_a");