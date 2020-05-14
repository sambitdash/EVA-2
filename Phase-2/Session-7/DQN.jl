using Flux
using Random: rand
using Flux: mse, cu
import Flux: update!
using StaticArrays

mutable struct ReplayMemory{NR, NS, L}
    m::MMatrix{NR, L, Float32}
    idx::Int
    n::Int
    ReplayMemory{NR, NS, L}() where {NR, NS, L} =
        new{NR, NS, L}(MMatrix{NR, L, Float32}(undef), 0, 0)
end

function Base.push!(mem::ReplayMemory{NR, NS},
                    last_state, state, last_reward, last_action) where {NR, NS}
    m, n, idx = mem.m, mem.n, mem.idx
    n < NR && (n += 1)
    idx = idx == NR ? 1 : idx+1
    println(last_state)
    m[idx, 1:NS] .= last_state
    m[idx, NS+1:2NS] .= state
    println(m[idx,:])
    m[idx, 2NS+1] = last_reward
    m[idx, end] = last_action
    mem.idx, mem.n = idx, n
end

function sample(mem::ReplayMemory{NR, NS}, sz::Int) where {NR, NS}
    n, m = mem.n, mem.m
    n < sz && error("Not enough data in memory for samples")
    ids = rand(1:n, sz)
    lstates = zeros(Float32, (NS, sz))
    states  = zeros(Float32, (NS, sz))
    rewards = zeros(Float32, sz)
    actions = zeros(Int, sz)
    for i = 1:n
        v = @view m[ids[i], :]
        lstates[:, i] .= v[1:NS]
         states[:, i] .= v[NS+1, 2NS]
        println("I am here")
        rewards[i] = v[end-1]
        actions[i] = Int(v[end])
    end
    return lstates, states, rewards, actions
end


mutable struct DQN{NS, NA}
    model
    optim
    memory::ReplayMemory
    gamma::Float32
    last_action::Int
    last_signal::Vector{Float32}
    last_reward::Float32
    function DQN{NS, NA}(gamma) where {NS, NA}
        model = Chain(Dense(NS, 16, relu),
                      Dense(16, 16, relu),
                      Dense(16, NA))
        return new(model,
                   ADAM(),
                   ReplayMemory{100000, NS, 2NS+2}(),
                   gamma,
                   1, zeros(Float32, NS), 0f0)
    end
end

select_action(dqn::DQN, s) = argmax(softmax(dqn.model(s)))

function learn!(dqn::DQN, csignals, nsignals, rewards, actionhb)
    model = dqn.model
    ps = params(model)
    gs = gradient(ps) do
        println(model)
        couts = model(csignals)
        println(csignals)
        couts = couts.*actionhb
        coutputs = reduce(+, couts, dims=1)
        println(coutputs)
        nouts = model(nsignals)
        println(size(nouts))
        nouts = nouts.*softmax(nouts, dims=1)
        targets = reduce(+, nouts, dims=1)
        println(size(targets))
        println(targets)
        value = mse(targets, coutputs)
        println("mse: ", value)
        return value
    end
    update!(dqn.optim, ps, gs)
end

function update!(dqn::DQN{NS, NA}, reward, signal) where {NS, NA}
    memory, last_signal, last_reward, last_action =
        dqn.memory, dqn.last_signal, dqn.last_reward, dqn.last_action

    push!(memory, last_signal, signal, last_reward, last_action)

    if memory.n <= 128 || rand() < 0.05
        action = rand(1:NA)
    else
        csignals, nsignals, rewards, actions = sample(memory, 128)
        actionhb = Flux.onehotbatch(actions, 1:NA)
        learn!(dqn, csignals, nsignals, rewards, actionhb)
        action = select_action(dqn, collect(signal))
    end
    dqn.last_action = action
    dqn.last_reward = reward
    copyto!(last_signal, signal)
    println(last_signal)
    return action
end
