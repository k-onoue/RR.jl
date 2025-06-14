# ############################################################################
#
#   Julia Implementation of GPR, TPRK and TPRD
#   (with selectable hyperparameter optimization)
#
# ############################################################################

# 将来的にはモジュールにしたほうが実験がスムーズかもしれない

using Optim
using LinearAlgebra
using SpecialFunctions
using Distributions
using Plots
using Distances

# --- カーネル関数 (共通) ---
struct SquaredExponentialKernel end

function kernel_matrix(::SquaredExponentialKernel, X::AbstractMatrix, params_θ::Vector{Float64})
    log_l, log_σ_n = params_θ
    l = exp(log_l)
    σ_n = exp(log_σ_n)
    sq_dist = pairwise(SqEuclidean(), X, dims=2)
    K_se = exp.(-0.5 * sq_dist / l^2)
    Σ = K_se + Diagonal(fill(σ_n^2 + 1e-6, size(X, 2)))
    return Σ
end

function kernel_matrix_grads(::SquaredExponentialKernel, X::AbstractMatrix, Σ::AbstractMatrix, params_θ::Vector{Float64})
    log_l, log_σ_n = params_θ
    l = exp(log_l)
    σ_n = exp(log_σ_n)
    n = size(X, 2)
    K_se = Σ - Diagonal(fill(σ_n^2 + 1e-6, n))
    sq_dist = pairwise(SqEuclidean(), X, dims=2)
    ∂Σ_∂log_l = K_se .* sq_dist / l^2
    ∂Σ_∂log_σ_n = Diagonal(fill(2 * σ_n^2, n))
    return [∂Σ_∂log_l, ∂Σ_∂log_σ_n]
end

# --- GPR / TP 構造体 ---
abstract type AbstractGPModel end

mutable struct GaussianProcess <: AbstractGPModel
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::SquaredExponentialKernel
    params_θ::Vector{Float64}

    function GaussianProcess(X, y; kernel=SquaredExponentialKernel(), init_params_θ=[0.0, 0.0])
        new(X, y, kernel, init_params_θ)
    end
end

mutable struct StudentTProcess <: AbstractGPModel
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::SquaredExponentialKernel
    model_type::Symbol
    params_θ::Vector{Float64}
    log_ν_minus_2::Float64

    function StudentTProcess(X, y, model_type; kernel=SquaredExponentialKernel(), init_params_θ=[0.0, 0.0], init_ν=5.0)
        @assert model_type in [:TPRK, :TPRD] "model_typeは :TPRK または :TPRD である必要があります。"
        @assert init_ν > 2 "νは2より大きくなければなりません。"
        new(X, y, kernel, model_type, init_params_θ, log(init_ν - 2))
    end
end

function get_ν(stp::StudentTProcess)
    return exp(stp.log_ν_minus_2) + 2
end

# --- モデルのインスタンスを生成するヘルパー関数 ---
# パラメータベクトルからモデルを生成
function build_model(model::GaussianProcess, p_full::Vector)
    return GaussianProcess(model.X, model.y; kernel=model.kernel, init_params_θ=p_full)
end
function build_model(model::StudentTProcess, p_full::Vector)
    return StudentTProcess(model.X, model.y, model.model_type; kernel=model.kernel, init_params_θ=p_full[1:2], init_ν=exp(p_full[3])+2)
end

# --- 主要な関数 (尤度、勾配、予測) ---

# GPR用
function log_marginal_likelihood(gp::GaussianProcess)
    X, y = gp.X, gp.y
    n = length(y)
    Σ = kernel_matrix(gp.kernel, X, gp.params_θ)
    C = cholesky(Σ)
    α = C \ y
    lml = -0.5 * dot(y, α) - 0.5 * logdet(C) - n/2 * log(2π)
    return lml
end

function gradient_log_marginal_likelihood(gp::GaussianProcess)
    X, y = gp.X, gp.y
    Σ = kernel_matrix(gp.kernel, X, gp.params_θ)
    C = cholesky(Σ)
    Σ⁻¹ = inv(C)
    α = Σ⁻¹ * y
    ∂Σ_∂θ_list = kernel_matrix_grads(gp.kernel, X, Σ, gp.params_θ)
    grad_params_θ = zeros(length(gp.params_θ))
    common_term = (α * α') - Σ⁻¹
    for i in 1:length(gp.params_θ)
        grad_params_θ[i] = 0.5 * tr(common_term * ∂Σ_∂θ_list[i])
    end
    return grad_params_θ
end

function predict(gp::GaussianProcess, X_new::AbstractMatrix)
    X, y = gp.X, gp.y
    Σ₁₁ = kernel_matrix(gp.kernel, X, gp.params_θ)
    k_params = [gp.params_θ[1], log(1e-9)]
    K₂₁ = kernel_matrix(gp.kernel, gp.X, X_new, k_params)'
    K₂₂_diag = diag(kernel_matrix(gp.kernel, X_new, k_params))
    Σ₁₁⁻¹y = Σ₁₁ \ y
    μ_pred = K₂₁ * Σ₁₁⁻¹y
    K̃₂₂_diag = K₂₂_diag - diag(K₂₁ * (Σ₁₁ \ K₂₁'))
    σ_pred = sqrt.(abs.(K̃₂₂_diag))
    return μ_pred, σ_pred
end

# TP用
function log_marginal_likelihood(stp::StudentTProcess)
    X, y, model_type = stp.X, stp.y, stp.model_type
    n = length(y)
    ν = get_ν(stp)
    Σ = kernel_matrix(stp.kernel, X, stp.params_θ)
    C = cholesky(Σ)
    α = C \ y
    β = dot(y, α)
    log_det_Σ = logdet(C)
    if model_type == :TPRK
        lml = -n/2 * log((ν - 2) * π) - 0.5 * log_det_Σ + (loggamma((ν + n) / 2) - loggamma(ν / 2)) - ((ν + n) / 2) * log(1 + β / (ν - 2))
    else # :TPRD
        lml = -n/2 * log(ν * π) - 0.5 * log_det_Σ + (loggamma((ν + n) / 2) - loggamma(ν / 2)) - ((ν + n) / 2) * log(1 + β / ν)
    end
    return lml
end

function gradient_log_marginal_likelihood(stp::StudentTProcess)
    X, y, model_type = stp.X, stp.y, stp.model_type
    n = length(y)
    ν = get_ν(stp)
    Σ = kernel_matrix(stp.kernel, X, stp.params_θ)
    C = cholesky(Σ)
    Σ⁻¹ = inv(C)
    α = Σ⁻¹ * y
    β = dot(y, α)
    ∂Σ_∂θ_list = kernel_matrix_grads(stp.kernel, X, Σ, stp.params_θ)
    grad_params_θ = zeros(length(stp.params_θ))
    if model_type == :TPRK
        common_term = ((ν + n) / (ν - 2 + β)) * (α * α') - Σ⁻¹
    else # :TPRD
        common_term = ((ν + n) / (ν + β)) * (α * α') - Σ⁻¹
    end
    for i in 1:length(stp.params_θ)
        grad_params_θ[i] = 0.5 * tr(common_term * ∂Σ_∂θ_list[i])
    end
    if model_type == :TPRK
        ∂lml_∂ν = -n / (2 * (ν - 2)) + 0.5 * (digamma((ν + n) / 2) - digamma(ν / 2)) - 0.5 * log(1 + β / (ν - 2)) + ((ν + n) * β) / (2 * (ν - 2)^2 + 2 * β * (ν - 2))
    else # :TPRD
        ∂lml_∂ν = -n / (2 * ν) + 0.5 * (digamma((ν + n) / 2) - digamma(ν / 2)) - 0.5 * log(1 + β / ν) + ((ν + n) * β) / (2 * ν^2 + 2 * β * ν)
    end
    grad_ν = ∂lml_∂ν * (ν - 2)
    return [grad_params_θ; grad_ν]
end

function predict(stp::StudentTProcess, X_new::AbstractMatrix)
    X, y, model_type = stp.X, stp.y, stp.model_type
    n = length(y)
    ν = get_ν(stp)
    Σ₁₁ = kernel_matrix(stp.kernel, X, stp.params_θ)
    k_params = [stp.params_θ[1], log(1e-9)]
    K₂₁ = kernel_matrix(stp.kernel, X, X_new, k_params)'
    K₂₂_diag = diag(kernel_matrix(stp.kernel, X_new, k_params))
    Σ₁₁⁻¹y = Σ₁₁ \ y
    μ_pred = K₂₁ * Σ₁₁⁻¹y
    β₁ = dot(y, Σ₁₁⁻¹y)
    K̃₂₂_diag = K₂₂_diag - diag(K₂₁ * (Σ₁₁ \ K₂₁'))
    if model_type == :TPRK
        scaling_factor = (ν + β₁ - 2) / (ν + n - 2)
    else # :TPRD
        scaling_factor = (ν + β₁) / (ν + n)
    end
    σ_pred = sqrt.(abs.(scaling_factor .* K̃₂₂_diag))
    return μ_pred, σ_pred
end

# 補助的なカーネル関数 (2つの入力を持つバージョン)
function kernel_matrix(k::SquaredExponentialKernel, X1::AbstractMatrix, X2::AbstractMatrix, params_θ::Vector{Float64})
    log_l, _ = params_θ
    l = exp(log_l)
    sq_dist = pairwise(SqEuclidean(), X1, X2, dims=2)
    return exp.(-0.5 * sq_dist / l^2)
end


# --- 選択的最適化のための fit! 関数 ---

function fit!(model::AbstractGPModel; fixed_params::Dict=Dict(), max_iters=100)
    # パラメータ名とインデックスのマッピング
    param_map = model isa GaussianProcess ? Dict(:l => 1, :sigma_n => 2) : Dict(:l => 1, :sigma_n => 2, :nu => 3)
    
    # 現在のモデルから全パラメータの初期値を取得
    if model isa GaussianProcess
        p_full = model.params_θ
    else
        p_full = [model.params_θ; model.log_ν_minus_2]
    end
    initial_p_full = copy(p_full)

    # 固定パラメータの値を設定
    for (key, val) in fixed_params
        idx = param_map[key]
        p_full[idx] = (key == :nu) ? log(val - 2) : log(val) # νはlog(ν-2)に、他はlogに変換
        initial_p_full[idx] = p_full[idx]
    end

    # 最適化対象のパラメータインデックスを決定
    optim_indices = setdiff(1:length(p_full), [param_map[k] for k in keys(fixed_params)])
    
    # 最適化対象がなければ何もしない
    if isempty(optim_indices)
        println("最適化対象のパラメータがありません。全てのパラメータは固定されています。")
        return
    end
    
    initial_p_optim = initial_p_full[optim_indices]

    # Optim.jlに渡す目的関数 (最適化対象の短いベクトル p_optim を受け取る)
    function objective(p_optim)
        p_full_temp = copy(initial_p_full)
        p_full_temp[optim_indices] = p_optim
        temp_model = build_model(model, p_full_temp)
        return -log_marginal_likelihood(temp_model)
    end

    # 勾配関数
    function gradient!(g, p_optim)
        p_full_temp = copy(initial_p_full)
        p_full_temp[optim_indices] = p_optim
        temp_model = build_model(model, p_full_temp)
        grad_full = -gradient_log_marginal_likelihood(temp_model)
        g[:] = grad_full[optim_indices]
    end

    # 最適化実行
    result = optimize(objective, gradient!, initial_p_optim, LBFGS(), Optim.Options(iterations=max_iters, show_trace=true))
    
    # 最適化後のパラメータでモデルを更新
    p_final = copy(initial_p_full)
    p_final[optim_indices] = Optim.minimizer(result)
    
    if model isa GaussianProcess
        model.params_θ = p_final
    else
        model.params_θ = p_final[1:2]
        model.log_ν_minus_2 = p_final[3]
    end
end