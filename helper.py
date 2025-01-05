import os
import pandas as pd
import numpy as np
import torch
import pathlib

import tensorflow as tf
import dask.dataframe as dd


def flatten_tensor_to_dataframe(tensor):
    # Ensure tensor is converted to NumPy (if it's a TensorFlow tensor)
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()  # Convert to NumPy array

    # Get the shape of the tensor (assuming [Learners, Questions, Attempts])
    num_learners, num_questions, num_attempts = tensor.shape

    # Initialize lists to store data
    student_ids = []
    question_ids = []
    question_attempts = []
    KC_Theoretical_Levels = []
    answer_scores = []  # This will store the Answer_Score from the tensor

    # Iterate through each combination of Student, Question, and Attempt
    for student_id in range(num_learners):
        for question_id in range(num_questions):
            for attempt in range(num_attempts):
                student_ids.append(student_id)
                question_ids.append(question_id)
                question_attempts.append(attempt + 1)  # Assuming 1-based attempt indexing
                KC_Theoretical_Levels.append("KC" + str(question_id))  # Adding KC level
                answer_scores.append(tensor[student_id, question_id, attempt])  # Answer_Score stored here

    # Create a DataFrame
    df = pd.DataFrame({
        "Student_Id": student_ids,
        "Question_Id": question_ids,
        "Question_Attempt": question_attempts,
        "Answer_Score": answer_scores,
        "KC_Theoretical_Levels": KC_Theoretical_Levels
    })

    # Drop NaN values in Answer_Score
    df_cleaned = df.dropna(subset=["Answer_Score"]).copy()  # Use .copy() to ensure it's not a view

    # Return the cleaned DataFrame and the dimensions of the tensor
    return df_cleaned, num_learners, num_questions, num_attempts


def read_as_tensor(args, set_parameters):
    global raw_T
    print("read_as_tensor")

    data_path = os.getcwd() + args.data_path[0] + args.data_path[1]

    # Initialize raw_T to a default value, such as None or an empty data structure.
    raw_T = None
    numpy_T = None

    if args.Course[0] == 'CSAL':
        print("Read CSAL dataset")
        lessonid = args.Lesson_Id[0]
        learning_stage = set_parameters['learning_stage']
        KCmodel = set_parameters['KCmodel']
        raw_T, numpy_T = read_CSAL_data(data_path, lessonid, learning_stage, KCmodel)

    if args.Course[0] == 'MATHia':
        print("MATHia path is {}".format(data_path))

        lessonid = args.Lesson_Id[0]
        learning_stage = set_parameters['learning_stage']
        KCmodel = set_parameters['KCmodel']
        Use_KC = set_parameters['Use_KC']

        raw_T, numpy_T = read_MATHia_data(data_path, lessonid, Use_KC)

    if args.Course[0] == 'ASSISMENTS':
        lessonid = args.Lesson_Id[0]
        learning_stage = set_parameters['learning_stage']
        KCmodel = set_parameters['KCmodel']
        Use_KC = set_parameters['Use_KC']

        raw_T, numpy_T = read_ASSISMENTS_data(data_path, lessonid, Use_KC)

    return raw_T, numpy_T


def read_CSAL_data(data_path, lessonid, learning_stage, KCmodel):
    df = pd.read_excel(data_path + '/' + '{}_alldata.xlsx'.format(lessonid), engine='openpyxl')

    level_data = df.query("Text_Difficulty=='{}'".format(learning_stage)).copy()

    level_data.loc[:, 'Question_Id'] = level_data['Question_Id'].astype('category').cat.codes
    level_data.loc[:, 'Student_Id'] = level_data['Student_Id'].astype('category').cat.codes

    cat_columns = level_data.select_dtypes(['category']).columns
    level_data.loc[:, cat_columns] = level_data[cat_columns].apply(lambda x: x.cat.codes)

    """Drop the columns"""
    level_data = level_data.drop(['Lesson_Id', 'IsCompleted', 'Text_Difficulty'], axis=1)

    if KCmodel == 'Unique':
        level_data['KC1'] = level_data['KC1'].astype(str)
        level_data['KC2'] = level_data['KC2'].astype(str)
        level_data['KC3'] = level_data['KC3'].astype(str)
        level_data['Question_Id'] = level_data['Question_Id'].astype(str)
        level_data['KC_Theoretical_Levels'] = level_data[['Question_Id', 'KC1', 'KC2', 'KC3']].agg('-'.join, axis=1)
        level_data['Question_Id'] = level_data['Question_Id'].astype(int)

    elif KCmodel == 'Single':
        level_data['KC1'] = level_data['KC1'].astype(str)
        level_data['KC2'] = level_data['KC2'].astype(str)
        level_data['KC3'] = level_data['KC3'].astype(str)
        level_data['KC_Theoretical_Levels'] = level_data[['KC1', 'KC2', 'KC3']].agg('-'.join, axis=1)

    level_data['KCindex'] = level_data['Student_Id'].astype(str) + "-" + level_data['KC_Theoretical_Levels']
    level_data = level_data.drop(['KC1', 'KC2', 'KC3'], axis=1)

    columns_list = list(['Student_Id', 'Question_Id', 'Question_Attempt', 'Answer_Score'])

    level_data = level_data[columns_list]

    raw_T, numpy_T = csal_to_tensor(level_data)  # students*questions*attempts,obs

    return raw_T, numpy_T


def read_MATHia_data(data_path, lessonid, Use_KC):
    print("read_MATHia_data")

    df = pd.read_csv(data_path + "/" + "{}.txt".format(lessonid), sep="\t")

    df = df.dropna(subset=['Step Name'])

    # print("Level (Workspace Id) include {}".format(df['Level (Workspace Id)'].nunique()))
    # print("Anon Student Id include {}".format(df['Anon Student Id'].nunique()))
    # print("KC Model(MATHia) include {}".format(df['KC (MATHia)'].nunique()))
    # print("Step Name include {}".format(df['Step Name'].nunique()))
    # print("Problem Name include {}".format(df['Problem Name'].nunique()))
    # print("Attempt At Step include {}".format(df['Attempt At Step'].nunique()))

    columns_list = list(
        ["Level (Workspace Id)", "Anon Student Id", "Problem Name", "KC (MATHia)", "Time", "Step Name",
         "Attempt At Step", "Outcome"])

    df = df[columns_list]
    # df = df.sort_values(by='Time',ascending=True)

    df['Answer_Score'] = df["Outcome"].map(lambda x: 1 if x == 'OK' else 0)
    df.loc[:, 'Student_Id'] = df['Anon Student Id'].astype('category').cat.codes
    df.loc[:, 'Question_Id'] = df['Step Name'].astype('category').cat.codes
    df['Question_Attempt'] = df['Attempt At Step']

    if Use_KC is True:
        df = df.drop(["Problem Name"], axis=1)
        df.rename(columns={"Step Name": "Problem Name"}, inplace=True)

    df = df.drop(columns_list, axis=1)
    df = df[['Student_Id', 'Question_Id', 'Question_Attempt', 'Answer_Score']]

    # Still have some problem on attempts on the problem steps.
    # print("df columns are {}".format(df.columns))

    raw_T, numpy_T = mathia_to_tensor(df)  # students*questions*attempts,obs

    return raw_T, numpy_T


def read_ASSISMENTS_data(data_path, lessonid, Use_KC):
    print("read_ASSISMENTS_data")

    new_df = pd.DataFrame()

    if lessonid == '2012-2013-data-with-predictions-4-final':
        np.random.seed(42)
        datafile = data_path + "/" + '2012-2013-data-with-predictions-4-final.csv'
        df = pd.read_csv(datafile)
        columns_list = (['problem_id', 'correct', 'attempt_count', 'skill_id', 'skill', 'user_id'])
        df = df[columns_list]

        # Identify 5% of the unique users
        selected_users = df['user_id'].drop_duplicates().sample(frac=0.05)
        # Filter out the selected users
        df = df[df['user_id'].isin(selected_users)]

        # Remove rows where the 'skill' column is NaN
        df = df.dropna(subset=['skill'])

        # Find the top 20 most frequent 'problem_id's
        top_problem_ids = df['problem_id'].value_counts().nlargest(20).index

        # Filter the DataFrame to include only these top 20 'problem_id's
        df = df[df['problem_id'].isin(top_problem_ids)]

        new_df['Student_Id'] = df['user_id']
        new_df['Question_Id'] = df['problem_id']
        new_df['Question_Attempt'] = df['attempt_count']
        new_df['Answer_Score'] = df["correct"]

        percentage = 99

    else:
        df = pd.read_csv(data_path + "/" + "{}.txt".format(lessonid), sep="\t")

        print("df column names are {}".format(df.columns))

        columns_list = list(['Anon Student Id', 'Step Name', 'Attempt At Step', 'Outcome', 'Problem Name'])
        df = df[columns_list]

        new_df['Student_Id'] = df['Anon Student Id'].astype('category').cat.codes
        new_df['Question_Id'] = df['Step Name'].astype('category').cat.codes
        new_df['Question_Attempt'] = df['Attempt At Step']
        new_df['Answer_Score'] = df["Outcome"].map(lambda x: 1 if x == 'CORRECT' else 0)

        print("new_df is {}".format(new_df.head()))

        percentage = 70

    raw_T, numpy_T = assisments_to_tensor(new_df, percentage)

    return raw_T, numpy_T


def mathia_to_tensor(df):
    num_learner = df.Student_Id.nunique()
    num_questions = df.Question_Id.nunique()

    # Attempts adjustment
    df, num_attempts = extractMaxAttempts(df, 95)

    raw_T = np.full((num_learner, num_questions, num_attempts), np.nan)

    df_numpy = df.to_numpy().astype(int)

    for (learner, question, attempt, obs) in df_numpy:
        raw_T[learner, question, attempt - 1] = obs

    return raw_T, df_numpy


def csal_to_tensor(level_data):
    num_learner = int(level_data.Student_Id.nunique())
    num_questions = int(level_data.Question_Id.nunique())
    num_attempts = int(checkMaxAttempts(np.unique(level_data.Question_Attempt)))

    raw_T = np.full((num_learner, num_questions, num_attempts), np.nan)
    level_data_numpy = level_data.to_numpy()

    for (learner, question, attempt, obs) in level_data_numpy:
        raw_T[learner, question, attempt] = obs

    return raw_T, level_data_numpy


def assisments_to_tensor(df, percentage):
    df, num_attempts = extractMaxAttempts(df, percentage)

    print("df columns names are {}".format(df.columns))

    df['Student_Id'] = df['Student_Id'].astype('category').cat.codes
    df['Question_Id'] = df['Question_Id'].astype('category').cat.codes

    num_learner = df.Student_Id.nunique()
    num_questions = df.Question_Id.nunique()

    print("num_learner is {}".format(num_learner))
    print("num_questions is {}".format(num_questions))
    print("num_attempts is {}".format(num_attempts))
    raw_T = np.full((num_learner, num_questions, num_attempts), np.nan)

    df_numpy = df.to_numpy().astype(int)

    print("raw_T is {}".format(raw_T.shape))

    for (learner, question, attempt, obs) in df_numpy:
        raw_T[learner, question, attempt - 1] = obs

    return raw_T, df_numpy


def extractMaxAttempts(df, percentage):
    frequency_percentages = df.Question_Attempt.value_counts(normalize=True) * 100
    cumulative_frequency = frequency_percentages.cumsum()
    attempts_within_percent = cumulative_frequency[cumulative_frequency <= percentage].index[-1]
    num_attempts = int(attempts_within_percent)

    df = df[df['Question_Attempt'] <= num_attempts]

    return df, num_attempts


def checkMaxAttempts(uniqueAttempts):
    count = len(uniqueAttempts)
    max = np.max(uniqueAttempts)

    if count > max:
        return count
    else:
        return max + 1


def calculate_sparsity(tensor):
    tensor = tensor.float()  # Convert to float, assuming tensor is a PyTorch tensor
    total_elements = tensor.numel()  # Get the total number of elements in the tensor
    nan_elements = torch.isnan(tensor).sum().item()  # Get the number of NaN elements as a numeric value
    sparsity = nan_elements / total_elements  # Perform the division
    return sparsity


def sparsity_level(tensor):
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.float32))
    total_elements = tf.size(tensor, out_type=tf.float32)
    sparsity = nan_count / total_elements
    return float(sparsity)


def sparsity_explore(tensor, numpy_T, mode):
    '''
    Function for exploring the sparsity of a tensor based on different pruning strategies.
    Modes:
    - 'attempt_wise': Prunes the tensor along its second dimension and calculates sparsity.
    '''
    print("sparsity_explore")

    if mode == "attempt_wise":
        print("mode is {}".format(mode))
        print("Attempt from numpy_T is {}".format(np.unique(numpy_T[:, 2])))

        prune_sparsity_dict = {}

        for i in range(tensor.shape[2]):
            prune_tensor = tensor[:, :, 0:(tensor.shape[2] - i)]

            keep_attempt = tensor.shape[2] - i
            numpy_T = numpy_T[numpy_T[:, 2] < keep_attempt]
            current_sparsity = sparsity_level(prune_tensor)
            prune_sparsity_dict[i] = {'prune_tensor': prune_tensor, 'prune_numpy_T': numpy_T,
                                      'sparsity': current_sparsity,
                                      'prune_slice_number': i}

        return prune_sparsity_dict


def save_tensor(tensor, set_parameters):
    results_dir = os.path.join(os.getcwd(), 'results', 'Ori_tensors')

    # Convert the tensor to tf.Example
    tf_example = tensor_to_tf_example(tensor)

    # Serialize to string
    tf_example_string = tf_example.SerializeToString()

    # print("tf_example_string is {}".format(tf_example_string))

    # Format the file name
    file_name = f"{set_parameters['course']}_{set_parameters['lesson_id']}_{set_parameters['learning_stage']}.tfrecord"

    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Join the directory path and the file name
    full_path = os.path.join(results_dir, file_name)

    # Write to TFRecord file
    with tf.io.TFRecordWriter(full_path) as writer:
        writer.write(tf_example_string)


def tensor_to_tf_example(tensor):
    tensor_np = tensor.numpy()

    # Flatten the tensor and convert it to a list of floats.
    # This also handles conversion of NaN values correctly.
    flat_tensor_list = tensor_np.flatten().tolist()
    tensor_shape = tensor_np.shape

    # Construct a feature dictionary
    features = {
        'tensor': tf.train.Feature(float_list=tf.train.FloatList(value=flat_tensor_list)),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=tensor_shape))
    }

    # Create a Features message using tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    # Use float_list instead of int64_list.
    return tf_example


def read_from_ori(data_path):
    global tensor
    # print("Data path is {}".format(data_path))

    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(parse_tfrecord)  # Parse the record into tensors

    # # Iterate through the dataset to print the shape of each tensor
    for tensor in dataset:
        print("Successful Extraction of Tensor")

    return tensor


def parse_tfrecord(example_proto):
    # Description of the data format
    feature_description = {
        'tensor': tf.io.VarLenFeature(tf.float32),
        'shape': tf.io.VarLenFeature(tf.int64)
    }

    # Parse the input tf.train.Example proto using the dictionary above
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Convert the tensor from a sparse format to a dense format
    tensor = tf.sparse.to_dense(example['tensor'])

    # Convert the shape from a sparse format to a dense format and cast it to int32
    shape = tf.sparse.to_dense(example['shape'], default_value=1)
    shape = tf.cast(shape, tf.int32)

    # Reshape the tensor to its original shape
    tensor = tf.reshape(tensor, shape)

    return tensor


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)