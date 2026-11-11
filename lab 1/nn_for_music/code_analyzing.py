import comet_ml
# TODO: ENTER YOUR API KEY HERE!! instructions above
COMET_API_KEY = "XV8FvjQvEZcZdYJvBlbIUyjJr"

# Import Tensorflow 2.0
import tensorflow as tf

# Download and import the MIT Introduction to Deep Learning package
# !pip install mitdeeplearning --quiet

import mitdeeplearning as mdl

# Import all remaining packages
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write


# TODO: !apt-get install abcmidi timidity > /dev/null 2>&1

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU

# TODO: assert len(tf.config.list_physical_devices('GPU')) > 0
assert COMET_API_KEY != "", "Please insert your Comet API Key"

# Download the dataset
songs = mdl.lab1.load_training_data()

# Print one of the songs_dataset to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)

# Convert the ABC notation to audio file and listen to it

# TODO: Я НЕ МОГУ ПРОСЛУШАТЬ ЕЁ, НО Я МОГУ ИСПОЛЬЗОВАТЬ ОНЛАЙН-ПРОИГРЫВАТЕЛИ АВС-НОТАЦИИ!!! В НИХ МОЖНО ВСТАВИТЬ ВЕСЬ ТЕКСТ, КОТОРЫЙ ВЫВОДИТСЯ print(example_song)
#mdl.lab1.play_song(example_song)

print()
# Join our list of song strings into a single string containing all songs_dataset
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
#   we can evaluate `char2idx["d"]`.
char2idx = {u:i for i, u in enumerate(vocab)}
#enumerate() — это функция, которая позволяет при итерации по последовательности получать сначала индекс элемента, потом сам элемент.
print("char2idx:", char2idx)
print()
# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)
print("idx2char:", idx2char)
print()


print('{')
for char,_ in zip(char2idx, range(83)): #с помощью zip в цикле используется первые 83 символа char2idx (в данном случае, все символы)
    #char - ключ словаря char2idx; _ - значение из range(83), которое мы игнорируем
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    #4s - s(строка) будет выведена в поле шириной не менее 4 символов (если строка короче, она будет дополнена пробелами).
    #3d - d(целое число) будет выведено как десятичное число, при этом поле вывода будет иметь минимальную ширину 3 символа.
    #repr() возвращает строковое представление символа, обычно заключённое в кавычки, что может помочь при отладке (например, различать пробелы, символы переноса и т.д.).
print('  ...\n}') #ставит многоточие, переходит на новую строку и в новой строке ставит "}".
#
# '''TODO: Write a function to convert the all songs_dataset string to a vectorized
#     (i.e., numeric) representation. Use the appropriate mapping
#     above to convert from vocab characters to the corresponding indices.
#
#   NOTE: the output of the `vectorize_string` function
#   should be a np.array with `N` elements, where `N` is
#   the number of characters in the input string
# '''
def vectorize_string(string):
  '''TODO'''
  return np.array([char2idx[char] for char in string])

vectorized_songs = vectorize_string(songs_joined)
print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"
#isinstance() — это встроенная функция, которая проверяет, принадлежит ли объект заданному типу данных или классу.

print(len(songs_joined), len(vectorized_songs))


### Batch definition to create training examples ###
# print(np.shape(vectorized_songs))
# print(tf.shape(vectorized_songs))
#Здесь tf.shape(tensor) возвращает тензор с формой (1,), содержащий значение [200679]
def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs_dataset string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size) #idx - массив случайных чисел принадлежащих [0, n-seq_length)
  #np.shape(idx) = (batch_size,).
  # кортеж из 1 элемента должен иметь запятую, чтобы он был отличим от выражения в скобках.
  # assert np.shape(idx) == (batch_size,)
  '''TODO: construct a list of input sequences for the training batch'''
  input_batch = [vectorized_songs[i:i+seq_length] for i in idx]

  '''TODO: construct a list of output sequences for the training batch'''
  output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  # np.reshape() меняет shape массива.
  return x_batch, y_batch


# Perform some simple tests to make sure your batch function is working properly!
test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
   print("======\n[FAIL] could not pass tests")
else:
   print("======\n[PASS] passed all tests!")
#
#
x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    #np.squeeze() удаляет измерения массива размером 1: [[[1], [2], [3]]] -> [1,2,3]
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))





def LSTM(rnn_units):
    # В LSTM 4 отдельных матрицы весов W взаимодействуют с входными данными (Wf, Wi, Wc, Wo). Эти 4 весовые матрицы объединяются в массив kernel.
    # В LSTM 4 отдельных матрицы весов U взаимодействуют со скрытым состоянием (Uf, Ui, Uc, Uo). Эти 4 весовые матрицы объединяются в массив recurrent_kernel.
  return tf.keras.layers.LSTM(
    rnn_units,
    return_sequences=True,
    recurrent_initializer='glorot_uniform',#recurrent_initializer инициализирует recurrent_kernel матрицу весов
    #kernel_initializer инициализирует kernel матрицу весов
    recurrent_activation='sigmoid',
    stateful=True,
  )


'''TODO: Add LSTM and Dense layers to define the RNN model using the Sequential API.'''
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    #embedding_dim - количество измерений embedding вектора
  model = tf.keras.Sequential([
    # tf.keras.Sequential() создаёт модель из последовательности слоёв, где один слой следует за другим

    # Layer 1: Embedding layer to transform indices into dense vectors
    #   of a fixed embedding size
    tf.keras.layers.Embedding(vocab_size, embedding_dim),

    # Layer 2: LSTM with `rnn_units` number of units.
    # TODO: Call the LSTM function defined above to add this layer.
    LSTM(rnn_units),

    # LSTM(rnn_units//2),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size.
    # TODO: Add the Dense layer.
    #'''TODO: DENSE LAYER HERE'''
    tf.keras.layers.Dense(vocab_size)
  ])

  return model

# Build a simple model with default hyperparameters. You will get the
#   chance to change these later.
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
model.build(tf.TensorShape([32, 100]))  # [batch_size, sequence_length]
#model.build() создаёт веса на основе указанной shape (формы) входных данных.


model.summary() #выводит основную информацию о модели, а именно:
#-Имя и тип всех слоев модели.
#-Выходная shape (форма) для каждого слоя.
#-Количество весовых параметров каждого слоя.
#-Общее количество обучаемых и необучаемых параметров модели.

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

#категориальное распределение - это дискретное распределение вероятностей,
# которое описывает возможные результаты случайной величины, которая может принимать одну из K возможных категорий,
# при этом вероятность каждой категории указывается отдельно.
sampled_indices = tf.random.categorical(pred[0], num_samples=1)
#tf.random.categorical() делает выборки из категориального распределения. pred.shape=(batch_size, sequence_length, vocab_size), но
#sampled_indices.shape()=(sequence_length, num_samples) = (sequence_length, 1)
print(sampled_indices)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#sampled_indices

print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
#idx2char[sampled_indices] - в массив можно передать массив, который будет считаться массивом индексов, но это не работает в классическом питоне.
#idx2char - np.ndarray, sampled_indices - np.ndarray, idx2char[sampled_indices] - np.ndarray.
#если:
#idx2char = np.array(['a', 'b', 'c', 'd', 'e'])
#sampled_indices = np.array([3, 0, 4])
#то idx2char[sampled_indices] = np.array(['d', 'a', 'e'])


'''TODO: define the loss function to compute and return the loss between
    the true labels and predictions (logits). Set the argument from_logits=True.'''
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) # TODO
  return loss

'''TODO: compute the loss using the true next characters from the example batch
    and the predictions from the untrained model several cells above'''
example_batch_loss = compute_loss(y, pred) # TODO

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
#numpy.mean() возвращает среднее значение элементов массива
#numpy() преобразует tensor объект в numpy.ndarray объект.


### Hyperparameter setting and optimization ###

vocab_size = len(vocab)

# Model parameters:
params = dict(
  num_training_iterations = 1000,  # Increase this to train longer
  batch_size = 16,  # Experiment between 1 and 64
  seq_length = 500,  # Experiment between 50 and 500
  learning_rate = 1e-2,  # Experiment between 1e-5 and 1e-1
  embedding_dim = 256,
  rnn_units = 1024,  # Experiment between 1 and 2048
)

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "analyzing.h5")
#os.path.join() формирует путь при помощи объединения различных путей, автоматически вставляя соответствующий разделитель
os.makedirs(checkpoint_dir, exist_ok=True)
#os.makedirs() создаёт папку



### Create a Comet experiment to track our training run ###

def create_experiment():
  # end any prior experiments

  if 'experiment' in locals():
    #locals() возвращает словарь с локальными переменными текущей области видимости (класса, функции, всей программы...)
    experiment.end()

  # initiate the comet experiment for tracking
  experiment = comet_ml.Experiment(
                  api_key=COMET_API_KEY,
                  project_name="MIT_Lab1_Part2_analysing")
  # log our hyperparameters, defined above, to the experiment
  for param, value in params.items():
    experiment.log_parameter(param, value)
    #experiment.log_parameter() записывает параметр в отчет по эксперименту comet ml
  experiment.flush()
  #experiment.flush() отправляет все накопленные данные эксперимента на сервер Comet
  return experiment






### Define optimizer and training operation ###

'''TODO: instantiate a new model for training using the `build_model`
  function and the hyperparameters created above.'''
model = build_model(len(vocab), params['embedding_dim'], params['rnn_units'], params['batch_size'])

'''TODO: instantiate an optimizer with its learning rate.
  Checkout the tensorflow website for a list of supported optimizers.
  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/
  Try using the Adam optimizer to start.'''
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
  # Use tf.GradientTape()
  with tf.GradientTape() as tape:

    '''TODO: feed the current input into the model and generate predictions'''
    y_hat = model(x)

    '''TODO: compute the loss!'''
    loss = compute_loss(y, y_hat)

  # Now, compute the gradients
  '''TODO: complete the function call for gradient computation.
      Remember that we want the gradient of the loss with respect all
      of the model parameters.
      HINT: use `model.trainable_variables` to get a list of all model
      parameters.'''
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the optimizer so it can update the model accordingly
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  #apply_gradients() обновляет значений обучаемых переменных модели на основе вычисленных градиентов.
  #есть zip(), потому что для apply_gradients() нужны пары (gradient, variable).

  return loss

##################
# Begin training!# TODO: ЧТОБЫ ТРЕНИТЬ РАСКОММЕНТИРУЙ
##################

# history = []
# plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
# #PeriodicPlotter() - создатель графика, обновляющегося через каждые 2 секунды
# experiment = create_experiment()
#
# if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
# #hasattr(tqdm, '_instances') возвращает True, если указанный объект(класс, ...) имеет указанный атрибут(объявленную переменную в классе, ...), в противном случае - False.
# #tqdm._instances.clear() удаляет все существующие прогресс-бары, если они существуют
#
# for iter in tqdm(range(params["num_training_iterations"])):
#   #tqdm() показывает прогресс-бар цикла(сколько итераций сделано и осталось, кол-во итераций в секунду)
#
#   # Grab a batch and propagate it through the network
#   x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
#   loss = train_step(x_batch, y_batch)
#
#   # log the loss to the Comet interface! we will be able to track it there.
#   experiment.log_metric("loss", loss.numpy().mean(), step=iter)
#   # Update the progress bar and also visualize within notebook
#   history.append(loss.numpy().mean())
#   plotter.plot(history)
#
#   # Update the model with the changed weights!
#   if iter % 100 == 0:
#     model.save_weights(checkpoint_prefix)
#
# # Save the trained model and the weights
# model.save_weights(checkpoint_prefix)
# experiment.flush()




'''TODO: Rebuild the model using a batch_size=1'''
model = build_model(len(vocab), params['embedding_dim'], params['rnn_units'], batch_size=1)

# Restore the model weights for the last checkpoint after training
model.build(tf.TensorShape([1, None]))
#batch_size=1, None обозначает, что длина последовательности может быть любой.
model.load_weights(checkpoint_prefix)

model.summary()


def generate_text(model, start_string, generation_length=1000):
  # Evaluation step (generating ABC text using the learned RNN model)

  '''TODO: convert the start string to numbers (vectorize)'''
  input_eval = [char2idx[i] for i in start_string] #shape() = (len(start_string),)
  input_eval = tf.expand_dims(input_eval, 0) #shape() = (1, len(start_string))
  #tf.expand_dims() добавляет дополнительное измерение тензору в указанной позиции.
  #tf.expand_dims() вставляет тензору 1 в массив shape() в индекс axis.
  #input_eval = tf.expand_dims(input_eval, 0). Это нужно сделать, так как модель ожидает вход с размерностью (batch_size, sequence_length)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  #model.reset_states()
  # Reset states of LSTM layers in the model
  for layer in model.layers:
    if isinstance(layer, tf.keras.layers.LSTM):
      #если слой - LSTM, то сбросить внутренние состояния слоя
      layer.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      '''TODO: evaluate the inputs and generate the next character predictions'''
      predictions = model(input_eval)

      # Remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      '''TODO: use a multinomial distribution to sample'''
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      #tf.random.categorical() возвращает тензор с shape(sequence_length, num_samples).
      #tf.random.categorical(predictions, num_samples=1)[-1,0] - первый элемент на последней строке => предсказанный на последнем временном шаге элемент.

      # Pass the prediction along with the previous hidden state
      #   as the next inputs to the model
      input_eval = tf.expand_dims([predicted_id], 0)
      #теперь input_eval имеет shape() = (1,1). Это можно передать модели, ожидающей shape() = (batch_size, sequence_length)
      '''TODO: add the predicted character to the generated text!'''
      # Hint: consider what format the prediction is in vs. the output
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

'''TODO: Use the model and the function defined above to generate ABC format text of length 1000!
    As you may notice, ABC files start with "X" - this may be a good start string.'''
generated_text = generate_text(model, start_string="X", generation_length=6000)


### Play back generated songs_dataset ###

generated_songs = mdl.lab1.extract_song_snippet(generated_text)
#mdl.lab1.extract_song_snippet() выводит список мелодий, найденных в сгенерированном тексте.

print(generated_songs)

for i in generated_songs:
  #это цикл для того чтобы вывести тексты песен так, чтобы можно было вставить в онлайн-проигрыватель
  for j in i:
    if j == "\n":
      print()
    else:
      print(j, end="")
  print()
  print()


# for i, song in enumerate(generated_songs):
#   # Synthesize the waveform from a song
#   waveform = mdl.lab1.play_song(song)
#
#   # If its a valid song (correct syntax), lets play it!
#   if waveform:
#     print("Generated song", i)
#     ipythondisplay.display(waveform)
#
#     numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
#     wav_file_path = f"output_{i}.wav"
#     write(wav_file_path, 88200, numeric_data)
#
#     # save your song to the Comet interface -- you can access it there
#     experiment.log_asset(wav_file_path)
#Этот код:
#Генерирует звуковые волны из списка generated_songs с помощью mdl.lab1.play_song(song).
#Проверяет, является ли песня допустимой (имеет правильный синтаксис).
#Воспроизводит песню, если она корректная, с помощью ipythondisplay.display(waveform).
#Конвертирует аудиоданные в числовой массив np.frombuffer(waveform.data, dtype=np.int16).
#Сохраняет аудиофайл output_{i}.wav с частотой дискретизации 88 200 Гц с помощью write(wav_file_path, 88200, numeric_data).
#Загружает файл в интерфейс Comet.ml для последующего анализа.

