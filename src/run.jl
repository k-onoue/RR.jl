using Dates, DataFrames, CSV, Statistics, Printf

# このスクリプト自身があるディレクトリ
const SRC_DIR = @__DIR__
# データセットディレクトリ
const DATASETS_DIR = joinpath(SRC_DIR, "..", "datasets")
# 結果保存ディレクトリ
const RESULTS_DIR = joinpath(SRC_DIR, "..", "results")

# モデル定義を含むファイルを読み込む
include(joinpath(SRC_DIR, "models.jl"))

"""
load_and_standardize_data(dataset_name, split_idx)

- dataset_name::String : データセット名
- split_idx::Int      : 0 から 9 の分割番号

返り値: (X_train, y_train_std, X_test, y_test_std)
"""
function load_and_standardize_data(dataset_name::String, split_idx::Int)
    base_path = joinpath(DATASETS_DIR, dataset_name, "split_$(split_idx)")

    # CSV 読み込み
    df_X_train = CSV.read(joinpath(base_path, "train_features.csv"), DataFrame, header=false)
    df_y_train = CSV.read(joinpath(base_path, "train_target.csv"),   DataFrame, header=false)
    df_X_test  = CSV.read(joinpath(base_path, "test_features.csv"),  DataFrame, header=false)
    df_y_test  = CSV.read(joinpath(base_path, "test_target.csv"),    DataFrame, header=false)

    # Matrix/Vector 変換（特徴量×サンプル の形に）
    X_train = Matrix(df_X_train)'
    y_train = vec(Matrix(df_y_train))
    X_test  = Matrix(df_X_test)'
    y_test  = vec(Matrix(df_y_test))

    # 標準化
    y_mean = mean(y_train)
    y_std  = std(y_train)
    y_train_std = (y_train .- y_mean) ./ y_std
    y_test_std  = (y_test  .- y_mean) ./ y_std

    return X_train, y_train_std, X_test, y_test_std
end

# RMSE（Root Mean Square Error）を計算するヘルパー
function rmse(y::AbstractVector, ŷ::AbstractVector)
    return sqrt(mean((y .- ŷ).^2))
end

function evaluate_models()
    dataset_names = ["Bike", "Concrete", "Diabetes", "ELE",
                     "Machine CPU", "MPG", "Neal", "Neal_XOutlier"]

    # 結果ディレクトリの作成
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
        println("`$(RESULTS_DIR)` ディレクトリを作成しました。")
    end

    # 集計用 DataFrame
    all_results_df = DataFrame(Dataset=String[], Model=String[],
                               Mean_RMSE=Float64[], Std_Dev_RMSE=Float64[])

    println("="^60)
    println("モデルパフォーマンス評価を開始します。")
    println("="^60)

    for dataset_name in dataset_names
        println("\n--- データセット: $(dataset_name) の評価 ---")

        # RMSE 格納用辞書
        results = Dict(
            :GPR => Float64[],
            :TPRK => Float64[],
            :TPRD => Float64[],
            :TPRK_fixed_nu => Float64[],
            :TPRD_fixed_nu => Float64[]
        )

        # モデル設定リスト
        model_configs = [
            (name=:GPR,           model_type=:GPR,  fixed_params=Dict()),
            (name=:TPRK,          model_type=:TPRK, fixed_params=Dict()),
            (name=:TPRD,          model_type=:TPRD, fixed_params=Dict()),
            (name=:TPRK_fixed_nu, model_type=:TPRK, fixed_params=Dict(:nu => 5.0)),
            (name=:TPRD_fixed_nu, model_type=:TPRD, fixed_params=Dict(:nu => 5.0))
        ]

        for i in 0:9
            println("  Split $(i)/9 を処理中...")
            try
                # データ読み込みと標準化
                X_train, y_train, X_test, y_test =
                    load_and_standardize_data(dataset_name, i)

                for config in model_configs
                    # モデル生成
                    if config.model_type == :GPR
                        model = GaussianProcess(X_train, y_train, init_params_θ=[0.0, log(0.1)])
                    else
                        model = StudentTProcess(
                            X_train, y_train, config.model_type,
                            init_params_θ=[0.0, log(0.1)], init_ν=5.0)
                    end
                    # フィッティング
                    fit!(model, fixed_params=config.fixed_params, max_iters=200)
                    # 標準化されたまま予測
                    μ_pred_std, _ = predict(model, X_test)
                    # RMSE 計算
                    error = rmse(y_test, μ_pred_std)
                    push!(results[config.name], error)
                end
            catch e
                println("    エラー発生: Split $(i) の処理をスキップします。エラー: ", e)
                for config in model_configs
                    push!(results[config.name], NaN)
                end
            end
        end

        # 結果表示・集計
        println("\n[結果] データセット: $(dataset_name)")
        println("-"^55)
        @printf("%-18s | %-18s | %-18s\n", "モデル", "平均RMSE", "RMSE標準偏差")
        println("-"^55)
        for config in model_configs
            name = config.name
            valid_rmses = filter(!isnan, results[name])
            mean_err = isempty(valid_rmses) ? NaN : mean(valid_rmses)
            std_err  = isempty(valid_rmses) ? NaN : std(valid_rmses)
            @printf("%-18s | %-18.4f | %-18.4f\n", string(name), mean_err, std_err)

            push!(all_results_df, (
                Dataset=dataset_name,
                Model=string(name),
                Mean_RMSE=mean_err,
                Std_Dev_RMSE=std_err
            ))
        end
        println("-"^55)
    end

    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    filename = "evaluation_summary_$(timestamp).csv"
    filepath = joinpath(RESULTS_DIR, filename)
    CSV.write(filepath, all_results_df)

    println("\n" * "="^60)
    println("全ての評価が完了しました。結果は $(filepath) に保存されました。")
    println("="^60)
end

# 直接実行時のみ
if abspath(PROGRAM_FILE) == @__FILE__
    evaluate_models()
end
