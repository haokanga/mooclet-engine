import pandas as pd


# Helper function for setting non-empty (key, value) pair in the given dictonary.
# Return True is there is a non-empty (key, value) pair.
def set_if_not_none(mapping, key, value):
    if value is not None:
        mapping[key] = value
        return True
    return False


# Relationship between version values and reward values are one-to-many.
# E.g. For each participant, one assigned version can map to many reward values. 
# But only the first (i.e. oldest) reward are considered valid to the assigned version.
def map_version_to_reward(
    values, 
    mooclet, 
    policy,  
    rewards, 
    variables,
    update_group=0,
    policy_params=None, 
    policy_params_history=None
):
    '''
    args:
        values: QuerySet object. Indicates the selected Value instances.
        mooclet: Model object. Indicates the selected Mooclet instance.
        policy: Model object. Indicates the selected Policy instance.
        rewards: QuerySet object. Indicates the reward in Variable instances.
        variables: QuerySet object. Indicates the selected Variable instances.
        update_group: Int. Indicates which update group for the datapoints.
        policy_params: Model object. Indicates the selected PolicyParameters 
            instances. Can be optional.
        policy_params_history: Model object. Indicates the selected 
            PolicyParametersHistory instances. Can be optional.
    
    returns:
        data: pandas.DataFrame object. Contains all data points for each reward 
            entry in datetime order (from oldest to newest).
    '''
    def without_keys(d, keys):
        return {x: d[x] for x in d if x not in keys}

    mooclet_name = mooclet.name
    policy_name = policy.name
    variable_names = list(variables.value_list('name', flat=True))
    reward_names = list(rewards.value_list('name', flat=True))

    # Create columns specified for the datapoints.
    columns = ["study", "learner", "arm_assign_time", "policy", "arm", "reward_name"] 
    columns += variable_names 
    columns += [ "reward_create_time"]

    # Invalid names in the policy parameter
    invalid_param_names = [
        "action_space", "outcome_variable", "contextual_variables", 
        "prior", "outcome_variable_name", "update_record"
    ]

    # Get parameterse from policy.
    # Note that parameters field may not exist in the policy, then parameters is set to None.
    parameters = None
    try:
        if policy_params_history:
            field = policy_params_history._meta.get_field(parameters)
            parameters = policy_params_history.parameters
        elif policy_params:
            field = policy_params._meta.get_field(parameters)
            parameters = policy_params.parameters
    except FieldDoesNotExist:
        pass

    # Add Coulmns for Policy parameters
    parameter_dict = without_keys(parameters, invalid_param_names) if parameters is not None else {}
    parameter_names = list(parameter_dict.keys()) if len(parameter_dict) != 0 else []
    columns += parameter_names
    columns += ["update_group"]

    # Initialize DataFrame with columns
    data = pd.DataFrame(columns=columns)

    value_df = pd.DataFrame.from_records(
        values.values(
            "mooclet__name",
            "learner__pk",
            "learner__name",
            "policy__name",
            "version__name",
            "version__version_json"
            "variable__name",
            "value",
            "timestamp"
        )
    ).rename(
        columns={
            "mooclet__name": "study",
            "learner__pk": "learner_id",
            "learner__name": "learner",
            "policy__name": "policy",
            "version__name": "arm",
            "version__version_json": "action_space",
            "variable__name": "name"
        }
    )

    version_df = value_df[value_df.name == "version"]
    reward_df = value_df[value_df.name.isin(reward_names)]
    context_df = value_df[value_df.name.isin(variable_names) & ~value_df.name.isin(reward_names)]

    version_df.reset_index(inplace=True)
    reward_df.reset_index(inplace=True)
    context_df.reset_index(inplace=True)

    record_df = pd.DataFrame(columns=["user_id"])
    if "update_record" in parameters and policy_params is None:
        record_df = pd.concat([record_df, pd.DataFrame(parameters["update_record"])])

    update_datapoint_count = 0
    for version_index, version_row in version_df.iterrows():
        datapoint_dict = {
            "study": version_row["study"],
            "learner": version_row["learner"],
            "arm_assign_time": version_row["timestamp"],
            "policy": version_row["policy"],
            "arm": version_row["arm"]
        }
        datapoint_dict.update(parameter_dict)

        action_space = dict(version_row["action_space"])

        # Mapping rewards
        time_range = reward_df["timestamp"] > datapoint_dict["arm_assign_time"]

        # Access next version row
        same_mooclet_next = version_df["study"] == datapoint_dict["study"]
        same_learner_next = version_df["learner"] == datapoint_dict["learner"]
        next_version_row = version_df[same_mooclet_next & same_learner_next].iloc[version_index + 1:].head(1)
        
        if len(next_version_row.index) != 0:
            end_time = reward_df["timestamp"] <= next_version_row["timestamp"]
            time_range &= end_time
        
        same_mooclet = reward_df["study"] == datapoint_dict["study"]
        same_learner = reward_df["learner"] == datapoint_dict["learner"]
        same_policy = reward_df["policy"] == datapoint_dict["policy"]
        same_version = reward_df["arm"] == datapoint_dict["arm"]
        
        mapped_rewards = reward_df[time_range & same_mooclet & same_learner & same_policy & same_version]
        reward_df = reward_df.drop(mapped_rewards.index)

        for reward_index, reward_row in mapped_rewards.iterrows():
            reward_datapoint = {
                "reward_name": reward_row["name"],
                "reward_create_time": reward_row["timestamp"],
                reward_row["name"]: reward_row["value"]
            }
        
            # Mapping contexts
            same_learner_context = context_df["learner"] == datapoint_dict["learner"]
            time_range_context = context_df["timestamp"] < datapoint_dict["arm_assign_time"]

            mapped_contexts = context_df.loc[time_range_context & same_learner_context]

            # Drop duplicates and keep the last one (i.e. closest to the reward create time)
            closest_contexts = mapped_contexts.drop_duplicates(subset=["name"], keep="last")
            for context_index, context_row in closest_contexts.iterrows():
                context_datapoint = {
                    context_row["name"]: context_row["value"]
                }
                reward_datapoint.update(context_datapoint)

            # Check if reward is used for updating parameters
            if update_datapoint_count < len(record_df.index):
                # Get all checking conditions
                check_update = record_df["user_id"] == version_row["learner_id"]
                check_update &= record_df[reward_row["name"]] == reward_row["value"]
                for action_name, action_value in action_space.items():
                    check_update &= record_df[action_name] == action_value
                
                if len(record_df[check_update].index) != 0:
                    # We have a updating datapoint
                    reward_datapoint["update_group"] = update_group
                    update_datapoint_count += 1
            
            reward_datapoint.update(datapoint_dict)
            data.append(reward_datapoint, ignore_index=True)
    
    # Return sorted datapoints by reward created time (from oldest to newest).
    data = data.sort_values(by='reward_create_time', ascending=True)
    data["reward_create_time"] = data["reward_create_time"].astype(str)
    data["arm_assign_time"] = data["arm_assign_time"].astype(str)
    data = data.assign(Index=range(len(data))).set_index('Index')

    return data


def request_data_by_variable(
    values, 
    variable,
    mooclet, 
    policy, 
    # versions,
    policy_params=None, 
    policy_params_history=None
):
    '''
    args:
        values: QuerySet object. Indicates the selected Value instances.
        variable: Model object. Indicates the selected Variable instance.
        mooclet: Model object. Indicates the selected Mooclet instance.
        policy: Model object. Indicates the selected Policy instance.
        # versions: QuerySet object. Indicates the selected Version instances.
        policy_params: Model object. Indicates the selected PolicyParameters instances.
            Can be optional. If optional, then policy_params_history should not be optional.
        policy_params_history: Model object. Indicates the selected PolicyParametersHistory
            instances. Can be optional. If optional, then policy_params should not be optional.
        learner: Model object. Indicates the selected Learner instance. Can be optional.
    
    returns:
        data: pandas.DataFrame object. Contains all data points for each reward 
            entry in datetime order (from oldest to newest).
    '''
    mooclet_name = mooclet.name
    variable_name = variable.name
    policy_name = policy.name
    # version_names = list(versions.value_list('name', flat=True))

    # Create columns specified for the datapoints.
    columns = ["study", "learner", "variable_created_time", "policy", "arm", variable_name]
