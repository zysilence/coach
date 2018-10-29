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
                self.state_space[observation_space_name] = ImageObservationSpace(
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
* Dropout & Batchnorm
    * 通过preset配置网络结构
        * 参考rl_coach/presets/CARLA_CIL.py
    * 通过preset参数直接配置
        * agent_params.network_wrappers['main'].input_embedders_parameters['observation'].dropout = True
            * 按照设计，通过该参数也可以，但是程序中写死了dropout_rate=0, dropdout不起作用
        * agent_params.network_wrappers['main'].input_embedders_parameters['observation'].batchnorm = True
* Reward clips
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
* 使用两个月数据做训练存在过拟合现象
    * 增加dropout层
    * 增加L2 norm
    * 使用两年数据，使用大网络
    
