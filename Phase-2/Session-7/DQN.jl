using Flux
using Random: rand
using Flux: mse, cu
import Flux: update!

mutable struct DQN
    model
    optim
    memory::Vector
    gamma::Float32
    last_action
    last_signal
    last_reward
    function DQN(ninput, naction, gamma)
        model = Chain(Dense(ninput, 16, relu),
                      Dense(16, 16, relu),
                      Dense(16, naction)) |> gpu
        return new(model, ADAM(), [], gamma, 1,
                   (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)
    end
end

select_action(dqn::DQN, s) = argmax(softmax(dqn.model(s)))

function learn!(dqn::DQN, csignals, nsignals, rewards, actions)
    ps = params(dqn.model)
    gs = gradient(ps) do
        couts = dqn.model(csignals)
        coutputs = gpu([couts[actions[i], i] for i = 1:length(actions)])
        nouts = dqn.model(nsignals)
        noutputs = gpu([maximum(nouts[:, i]) for i = 1:size(nouts, 2)])
        targets = gpu(map(noutputs, rewards) do op, r
            if abs(r - 1f0) < 1f-6 || abs(r + 1f0) < 1f-6
                return r
            else
                return dqn.gamma*op + r
            end)
        end
        
        value = gpu(mse(targets, coutputs))
        println("mse: ", value)
        return value
    end
    update!(dqn.optim, ps, gs)
end

function update!(dqn::DQN, reward, signal)
    memory, last_signal, last_reward, last_action =
        dqn.memory, dqn.last_signal, dqn.last_reward, dqn.last_action

    push!(memory, (last_signal, signal, last_reward, last_action))
    lmem = length(memory)
    lmem > 100000 && popfirst!(memory)
    if lmem > 128 && rand() > 0.05
        batch = rand(memory, 128)

        csigns, nsigns, rewards, actions = collect(zip(batch...))
       
        csignals = gpu([ Float32(csigns[j][i]) for i=1:length(csigns[1]), j=1:length(csigns)])
        
        nsignals = gpu([ Float32(nsigns[j][i]) for i=1:length(nsigns[1]), j=1:length(nsigns)])
        rewards  = gpu(collect(rewards))
        actions  = gpu(collect(actions))

        learn!(dqn, csignals, nsignals, rewards, actions)
        action = select_action(dqn, collect(signal))
    else
        action = rand(1:4)
    end
    dqn.last_action = action
    dqn.last_signal = signal
    dqn.last_reward = reward
    return action
end
