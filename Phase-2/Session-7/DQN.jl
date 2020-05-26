using Flux
using Random: rand
using Flux: mse, cu
import Flux: update!
using StaticArrays
using LinearAlgebra

mutable struct ReplayMemory{NR, NS, L}
    m::MMatrix{L, NR, Float32}
    idx::Int
    n::Int
    ReplayMemory{NR, NS, L}() where {NR, NS, L} =
        new{NR, NS, L}(MMatrix{L, NR, Float32}(undef), 0, 0)
end

function Base.push!(mem::ReplayMemory{NR, NS},
                    last_state, state, last_reward, last_action) where {NR, NS}
    m, n, idx = mem.m, mem.n, mem.idx
    n < NR && (n += 1)
    idx = idx == NR ? 1 : idx+1

    m[1:NS, idx]     .= last_state
    m[NS+1:2NS, idx] .= state
    m[end-1, idx] = last_reward
    m[end, idx]   = last_action

    mem.idx, mem.n = idx, n
end

function sample(mem::ReplayMemory{NR, NS}, sz::Int) where {NR, NS}
    n, m = mem.n, mem.m
    n < sz && error("Not enough data in memory for samples")
    ids = rand(1:n, sz)
    lstates = zeros(Float32, (NS, sz))
    states  = zeros(Float32, (NS, sz))
    rewards = zeros(Float32, (1,  sz))
    actions = zeros(Int, sz)
    for i = 1:sz
        v = @view m[:, ids[i]]
        lstates[:, i] .= v[1:NS]
         states[:, i] .= v[NS+1:2NS]
        rewards[1, i] = v[end-1]
        actions[i] = Int(v[end])
    end
    return lstates, states, rewards, actions
end


mutable struct DQN{NS, NA}
    model
    modelb
    optim
    memory::ReplayMemory
    gamma::Float32
    last_action::Int
    last_signal::Vector{Float32}
    last_reward::Float32
    trained::Bool
    update_step::Int
    function DQN{NS, NA}(gamma) where {NS, NA}
        model  = Chain(Dense(NS, 64, relu),
                       Dense(64, 64, relu),
                       Dense(64, NA)) |> gpu
        modelb = Chain(Dense(NS, 64, relu),
                       Dense(64, 64, relu),
                       Dense(64, NA)) |> gpu
        return new(model,
                   modelb,
                   ADAM(),
                   ReplayMemory{100000, NS, 2NS+2}(),
                   gamma,
                   1, zeros(Float32, NS), 0f0,
                   false,
                   1)
    end
end

function select_action(dqn::DQN{NS, NA}, s) where {NS, NA}
    vals = softmax(dqn.model(s))
    ix, v = 1, vals[1]
    for i = 2:NA
        if v < vals[i]
            v = vals[i]
            ix = i
        end
    end
    return ix
end

function learn!(dqn::DQN{NS, NA}, csignals, nsignals, rewards, actionhb) where {NS, NA}
    model, modelb, gamma = dqn.model, dqn.modelb, dqn.gamma
    ps = params(model)
    ones_arr = ones(Float32, (1, NA)) |> gpu
    gamma_arr = fill(gamma, (1, NA+1)) |> gpu
    gamma_arr[1, end] = 1f0
    sz = size(rewards, 2)
    mask = ones(Float32, (NA, sz)) |> cpu
    for i=1:sz
        if abs(rewards[1, i]-1f0) < 1f-6 || abs(rewards[1, i]+1f0) < 1f-6
            mask[:, i] .= 0
        end
    end
    mask = gpu(mask)
    nouts = modelb(nsignals)
    nouts = nouts.*softmax(nouts, dims=1)
    nouts = mask.*nouts
    nouts = vcat(nouts, rewards)
    targets = gamma_arr*nouts
    gs = gradient(ps) do
        couts = model(csignals)
        couts = couts.*actionhb
        coutputs = ones_arr*couts
        v1, v2 = mse(targets, coutputs), 10sum(norm, ps)
        value = v1 + v2
        println("mse: ", value, " v1: ", v1, " v2: ", v2)
        return value
    end
    update!(dqn.optim, ps, gs)
    #if dqn.update_step % 16 == 0
        Flux.loadparams!(modelb, ps)
    #end
    dqn.update_step += 1
end

function update!(dqn::DQN{NS, NA}, reward, signal, episode=:c) where {NS, NA}
    memory, last_signal, last_action = dqn.memory, dqn.last_signal, dqn.last_action

    episode === :b ||
        push!(memory, last_signal, signal, reward, last_action)

    if dqn.memory.n > 128
        csignals, nsignals, rewards, actions = sample(memory, 128) |> gpu
        actionhb = Flux.onehotbatch(actions, 1:NA) |> gpu
        learn!(dqn, csignals, nsignals, rewards, actionhb)
    end
    
    action = rand() < 0.05 ? rand(1:NA) :
        select_action(dqn, gpu(collect(signal)))

    if episode !== :e
        dqn.last_action = action
        copyto!(last_signal, signal)
    end
    return action
end
