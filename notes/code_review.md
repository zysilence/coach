# Config
## Environment config
* Config file: "coach/rl_coach/environments/user/config/config.json"
* DATA
    * test_start_date
        * 开始的window_size个steps是没有结果的，因为这些steps对应的数据的历史数据不全，所以实际测试的是
        DATA.test_end_date - DATA.test_start_date - STATE.step_window个step的值，可以设置DATA.test_start_date提前step_window个时间单位
# STATE
* step_window
    * 设置值越大，看到的历史信息越多，更能对趋势建模, 比如step_window=90要好于step_window=30
* 特征
    * 是否需要对各个维度的特征做Normalization，参考Facebook开源的RL框架Horizon
    
# Netowrk
## Input Embedders
* CNN or Dense?
    * "get_input_embedder()" in "rl_coach/architectures/tensorflow_components/general_network.py"
        ```python
        type = "vector"
        if isinstance(allowed_inputs[input_name], PlanarMapsObservationSpace):
            type = "image"

        ```
        * 该语句决定了网络的类型
        * type(self.spaces.state.sub_spaces['observation'])等于ImageObservationSpace
        * ImageObservationSpace 是 PlanarMapsObservationSpace的子类
    * "LevelManager.build()" in "rl_coach/level_manager.py"
      ```python
      spaces = SpacesDefinition(state=self.real_environment.state_space,
                                  goal=self.real_environment.goal_space,  # in HRL the agent might want to override this
                                  action=action_space,
                                  reward=self.real_environment.reward_space)

      ```
        * spaces由环境决定; self.real_environment.state_space.sub_space_observation定义了类型是ImageObservationSpace
    * "rl_coach/environments/gym_environment.py" 
      ```python
      for observation_space_name, observation_space in state_space.items():
          if len(observation_space.shape) == 3 and observation_space.shape[-1] == 3:
              # we assume gym has image observations which are RGB and where their values are within 0-255
              se旦训练好以后，模型会比较难ovelf.state_space[observation_space_name] = ImageObservationSpace(
                  shape=np.array(observation_space.shape),
                  high=255,
                   channels_axis=-1
              )
          else:
              self.state_space[observation_space_name] = VectorObservationSpace(
                  shape=observation_space.shape[0],
                  low=observation_space.low,
                  high=observation_space.high
      ``` 
      当输入为3维，且第三维只有三个元素时，认为是ImageObservationSpace
    * 总结
        * environment的定义在'rl_coach/environments'目录，如gym环境的定义为"rl_coach/environments/gym_environment.py'
        * environment决定state_space('observation' or 'measurement')的空间类型
            * VectorObservationSpace
            * ImageObservationSpace
            * PlanarMapsObservationSpace
            * 其中，ImageObservationSpace是PlanarMapsObservationSpace的子类
        * agent--> create_networks --> NetworkWrapper() --> GeneralTensorFlowNetwork() --> get_input_embedder():
            * Defined in 'rl_coach/architectures/tensorflow_components/general_network.py'
            * 判断输入的状态空间类型：
                * 空间类型为PlanarMapsObservationSpace，则input_embedder的类型为'image', 否则为'vector'
                * 依据input_embedder类型加载对应的embedder类：
                ```python
                embedder_path = 'rl_coach.architectures.tensorflow_components.embedders.' + embedder_params.path[type]
                ```
        * 若个性化定制不同的网络结构, 可以在以上两个地方修改
* 可否使用autoencoder自动提取特征？
    * 参考[AlphaAI项目](https://github.com/VivekPa/AlphaAI)
    
## Dropout & Batchnorm
* 通过preset配置网络结构
    * 参考rl_coach/presets/CARLA_CIL.py
* 通过preset参数直接配置
    * agent_params.network_wrappers['main'].input_embedders_parameters['observation'].dropout = True
        * 按照设计，通过该参数也可以，但是程序中写死了dropout_rate=0, dropdout不起作用
    * agent_params.network_wrappers['main'].input_embedders_parameters['observation'].batchnorm = True
## Reward clips
*  btc训练中看到了很高的reward，尝试下reward clips。在preset中的配置如下：
```python
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.environments.gym_environment import GymEnvironmentParameters
  
input_filter = InputFilter(is_a_reference_filter=True)
input_filter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
env_params = GymEnvironmentParameters()
env_params.default_input_filter = input_filter
```

# 问题
## 过拟合
* 使用两个月数据做训练存在过拟合现象
    * 增加dropout层
    * 增加L2 norm
    * 使用两年数据，使用大网络
    
## 训练时随机挑选episode的开始，会不会导致有的没有采样过 

## 交易手续费问题
* 避免频繁交易，采用大的时间尺度作为最小交易单元，如4H, 1D等；通过实验证明1H在训练集上盈利，测试集上亏损

## 如何避免在熊市开仓？
* 增加判别牛市或熊市的模块,如一个分类器
* 设计reward函数，对熊市有更大的惩罚

## DQN模型的不稳定性
* 训练的某个时刻，在测试集上性能很好，多训练一段时间，测试集上性能变差，即测试集上性能时好时坏
* 考虑使用Double-DQN, Dueling-DQN, Double-Dueling-DQN等算法提高稳定性
