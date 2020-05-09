using FileIO
using Blink
using LinearAlgebra

include("DQN.jl")

struct Car
    img::Matrix
    sensors::NTuple{6, Tuple{Float32, Float32}}
    function Car()
        img = load("car.png")
        l, w = size(img)
        sd = 5f0
        sensors = ((-w/2-sd,  -sd), (0,   -sd), (w/2+sd,  -sd),
                   (-w/2-sd, l+sd), (0,  l+sd), (w/2+sd, l+sd))
        return new(img, sensors)
    end
end

struct CarState
    loc::Tuple{Float32, Float32}
    goal::Tuple{Float32, Float32}
    ang::Float32 #Clockwise positive
    
end
CarState()=CarState((0f0, 0f0), (890f0, 890f0), 0f0)
function signal(s::CarState)
    v = s.goal .- s.loc
    println(v)
    return v..., s.ang
end

function dist2(s::CarState)
    v = s.goal .- s.loc
    return dot(v, v)
end

ANG_STEP = 10f0

function rotate(s::CarState, angle)
    ang = s.ang + angle
    while ang < -pi
        ang += 2pi
    end
    while ang > pi
        ang -= 2pi
    end
    return CarState(s.loc, s.goal, ang)
end

left(s::CarState)  = rotate(s, -pi/180f0*ANG_STEP)
right(s::CarState) = rotate(s,  pi/180f0*ANG_STEP)

forward(s::CarState, d) = CarState(s.loc .+ (-d*sin(s.ang), d*cos(s.ang)),
                                   s.goal, s.ang)

function get_curr_loc(s::CarState, v)
    sth, cth, ox, oy = sin(s.ang), cos(s.ang), s.loc[1], s.loc[2]
    return v[1]*cth - v[2]*sth + ox, v[1]*sth + v[2]*cth + oy
end

get_curr_sensors(c::Car, s::CarState) = get_curr_loc.(Ref(s), c.sensors)

FWD_SPEED = 500f0                        # Pixels per sec
BK_SPEED_FACTOR = 5f0                   # Ratio of forward and backward speed
BK_SPEED  = FWD_SPEED / BK_SPEED_FACTOR  
ACTION_EVAL_INTERVAL = 10f0/FWD_SPEED     # Average check every 10 fwd pixels
ACTIONS = (:forward, :back, :left, :right)

map = Matrix{Bool}(load("maze.png"))

l_m, w_m = size(map)
println(l_m, " ", w_m)

w = Window()
loadfile(w, joinpath(@__DIR__, "file.html"))

goal = (840f0, 840f0)
bloc = (50f0, 50f0)
episode = 0

car, cs = Car(), CarState(bloc, goal, 0f0)

dqn = DQN(3, 4, 0.9)
last_reward = 0f0

inited = false
tm = time()

valid_sensor(sensor) =
    (global l_m, w_m; 1 <= sensor[1] <= w_m && 1 <= sensor[2] <= l_m)

sensor_on_wall(sensor) = (global map;
                          !map[round(Int, sensor[2]), round(Int, sensor[1])])

println(:none, " ", cs)


function reward(ps::CarState, s::CarState, a)
    global car
    sensors = get_curr_sensors(car, s)
    for sensor in sensors
        valid_sensor(sensor) || return (-1f0, true)
        sensor_on_wall(sensor) && return (-1f-1, false)
    end
    d = dist2(s)
    d < 100f0 && return (1f0, true)
    pd = dist2(ps)
    d < pd && return (1f-1, false)
    return (-1f-2, false)
end

function eval_action(timer)
    global inited, cs, car, w, bloc, goal, tm, episode, last_reward, dqn

    try 
        if !inited
            episode += 1
            cs = CarState(bloc, goal, 0f0)
            @js w carview.drawp(0, $(bloc...), true)
            inited = true
            return
        end

        action = ACTIONS[update!(dqn, last_reward, signal(cs))]
        println(action)
        speed = action == :back ? -BK_SPEED : FWD_SPEED
        dtm = 10f-2
        d = speed*dtm
        println(d)
        ns = cs

        action == :left  && (ns = left(ns))
        action == :right && (ns = right(ns))
        if action == :forward || action == :back
            ns = forward(ns, d)
        end

        r, isterminal = reward(cs, ns, action)

        println(r, " ", isterminal)
        if isterminal
            inited = false
            return
        end
    
        cs = ns

        println("episode = $episode ", action, " ", cs, " ", d)

        ang = action == :left  ? -ANG_STEP :
              action == :right ?  ANG_STEP : 0
        yd  = (action == :forward || action == :back) ? d : 0
        println(ang, " ",  yd)
        @js w carview.drawp($ang, 0, $yd)
    catch e
        println(e)
        throw(e)
    finally
    end
end

t = Timer(eval_action, 1, interval=ACTION_EVAL_INTERVAL)
wait(t)
read(stdin, 1)
close(t)
