using Flux
using Random: rand
using Flux: mse
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
                      Dense(16, naction))
        return new(model, ADAM(), [], gamma, 1, (0, 0, 0), 0)
    end
end

select_action(dqn::DQN, s) = argmax(softmax(dqn.model(s)))

function learn!(dqn::DQN, csignals, nsignals, rewards, actions)
    ps = params(dqn.model)
    gs = gradient(ps) do
        couts = dqn.model(csignals)
        coutputs = [couts[actions[i], i] for i = 1:length(actions)]
        println(size(coutputs))
        nouts = dqn.model(nsignals)
        noutputs = [maximum(nouts[:, i]) for i = 1:size(nouts, 2)]
        println(size(noutputs))
        targets  = dqn.gamma*noutputs .+ rewards
        value = mse(targets, coutputs)
        println(value)
        return value
    end
    update!(dqn.optim, ps, gs)
end

function update!(dqn::DQN, reward, signal)
    memory, last_signal, last_reward, last_action =
        dqn.memory, dqn.last_signal, dqn.last_reward, dqn.last_action

    push!(memory, (last_signal, signal, last_reward, last_action))
    if length(memory) > 100
        batch = rand(memory, 100)

        csigns, nsigns, rewards, actions = collect(zip(batch...))
       
        csignals = [ csigns[j][i] for i=1:length(csigns[1]), j=1:length(csigns)]
        nsignals = [ nsigns[j][i] for i=1:length(nsigns[1]), j=1:length(nsigns)]
        rewards  = collect(rewards)
        actions  = collect(actions)

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
