# [Computer Graphics]()

## 第一章 基础知识

- 计算机图形的类型：

  - 按绘制方式分：
    - **Raster Graphics**（光栅图形技术）
      - 类型：JPEG, PNG, APNG, GIF
    - **Vector Graphics**（向量图形技术）
      - 类型：SVG, SVG, EPS, EPS, PDF, PDF, AI
  - 按控制类型分：
    - **Interactive Graphics**（交互式图形）
      - 例如：video game
    - **Passive Graphics**（被动图形）
      - 例如：TV show

- 计算机图形的用途：

  1. **Computer-aided Design**（计算机辅助设计）
     - Automobiles，Aircraft，multi-window environment，Real-time animations，lighting models

  2. **Presentation Graphics**（展示图）
     - Bar charts，Pie charts，Line graphs，Surface graphs

  3. **Computer Art**（计算机艺术）
     - Cartoon drawing，watercolor painting

  4. **Entertainments**（娱乐）
     - Movies，Short films，TV serials

  5. **Education and Training**（教育）
     - Flight simulator，simulators of military tank

  6. **Image Processing**（图片处理）
     - Enhance the quality of a picture

  7. **Graphical User Interface GUI**（用户图形界面）
     - A GUI uses windows, icons, and menus to carry out commands

  8. **Visualization**（可视化）
     - Scientific visualization
     - Non-scientific visualization

- 图形系统的组成：

  - **Input devices**： 

    - **Manual data entry devices**：Key-board, mouse
    - **Direct data entry devices**：Scanner，OCR

  - **Output devices**： 

    - Printers：
      - Impact Printers：通过物理接触打印
        1. Daisy Wheel Printers
        2. Drum Printers
        3. Dot Matrix Printer

      - Non-impact Printers：不通过物理接触打印
        1. Inkjet Printer
        2. Laser Printer

      - Plotters：
        1. Flatbed Plotter
        2. Drum Plotter

  - **Display devices**： 

    1. Cathode-Ray Tube（CRT）
       - 扫描方式：
          - Raster Scan：从上到下全部扫描
          - Vector scan：只扫描更改的部分
       - 组成部分：
         - Electron Gun
         - Focusing and Accelerating Anodes
         - Horizontal and Vertical Deflection Plates
         - Phosphorus-coated Screen
    2. Color CRT Monitor
       - Beam-Penetration Method
       
        - Shadow-Mask Method
    3. Liquid crystal display（LCD）
    4. Light Emitting Diode（LED）
    5. Direct View Storage Tubes（DVST）
       - Primary Gun：It is used to store the picture information
       - Flood Gun：It is used to display a picture on the screen
    6. Plasma Display
    7. 3D Display

  - **Interfacing devices**： video input output to a system or interface to a Television


## 第二章 图形系统

- **Color Model**（颜色模型）

  - Additive Color Model（RGB）
  - Subtractive Color Model（CMYK）: Cyan, Magenta, Yellow, and Black

- **Image Representation**（图像表示）

  - 表示方法：数字图像被定义为二维阵列，图像元素以行和列的形式排列到图像区域。
  - 像素pixel：是图片在电脑上显示的最小单位
  - 分辨率：图片像素的个数

- **Format of Image Files**（图片格式）

  - **JPEG**：用于储存摄影图片
  - **PNG**：用于储存网络图片
  - **GIF**：储存的颜色多达256种
  - **TIFF/ TIF**：用于专业摄影

- 二维图像变换：

  - 2D Translation

  - 2D Scaling

  - 2D Rotation
    $$
    Anti-Clockwise \ Rotation\begin{cases}P_1 = P_0×cos θ - Q_0×sin θ\\
    Q_1 = P_0×sin θ + Q_0×cos θ\end{cases}\\
    Clockwise \ Rotation\begin{cases}P_1 = P_0×cos θ + Q_0×sin θ\\
    Q_1 = P_0×sin θ + Q_0×cos θ\end{cases}
    $$

  - 2D Reflection



## 第三章 图形绘制

- **Output primitives**（输出基本图形）

  - **Points**

  - **Lines**

  - **Polyline**（折线）

  - **polygon**（多边形）

  - **Filled region**
    - Seed Fill Algorithm（种子填充算法）
      1. Flood filled
      2. Boundary filled
    - Scan Line Algorithm（扫描线算法）

- **Line Drawing Algorithm**（画线算法）

  - **Digital Differential Analyzer**（DDA）
    $$
    (x_{k+1}, y_{k+1}) = 
    \begin{cases} 
    (x_k + 1, y_k + m)\ \ \ \ \ \  m <=1\\\\
    (x_k + \frac{1}{m}, y_k + 1)\ \ \ \ \ \  m >1
    \end{cases}
    $$

  - **Bresenham’s Line Drawing algorithm**
    $$
    p_1 = 2∆y- ∆x\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \\
    \ \ \ \ \ \ \ \  p_{k+1} = \begin{cases} 
    p_k + 2∆y-2∆x\ \ \ \ \ \  p_k>= 0\\\\
    p_k + 2∆y\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  p_k <0
    \end{cases}\\
    (x_{k+1}, y_{k+1}) = 
    \begin{cases} 
    (x_k + 1, y_k + 1)\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  p_k>= 0\\\\
    (x_k + 1, y_k)\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  p_k <0
    \end{cases}\\
    $$

  - **Mid point line drawing Algorithm**
    







# [Audio and Speech Processing]()

## Chapter 1: Introduction

- Audio capture and sampling, Speech analysis, Speech coding, Speech recognition.

## Chapter 2: From Analog to Digital Sampling

- Types:

  - The phonograph（留声机&唱片）

    - It was difficult to keep records scratch free which produced popping noises（很难让记录保持无刮痕，这会产生爆裂声）

    - Another phenomenon was groove lock which caused  sections to repeat over and over again.（另一种现象是音槽锁定，这会导致某些段落反复循环播放）

    - Records could warp if left close to a heat source making them unplayable.（如果靠近热源，记录可能会扭曲，使其无法播放）

  

    - Tape&Cassette（磁带）

      - Cassette tapes have a linear playback mechanism, making it difficult to locate specific tracks or sections.（磁带采用线性播放机制，很难快速定位特定的曲目或片段）


  - Soundcard（声卡）

    - *Sample rate*, and *number of bits*![image-20250106214109526](C:\Users\Crazbeans\Desktop\HappyThing\音频处理\image-20250106214109526.png)

      

      ![image-20250107224537214](C:\Users\Crazbeans\Desktop\HappyThing\音频处理\image-20250107224537214.png)


- Clip methods:
  - Hard Clip（硬剪辑）：The portions of a signal that exceed a certain threshold are cut off directly.
  - Soft Clip（软剪辑）：The signal is gradually compressed as it approaches the threshold.
- Basic concepts:

  - **Sample rate**（采样率）： This is the number of samples of audio signal taken every second by the analog to digital converter in the sound card device.
  - **Shannon's theorem**（香农定理）：Any signal can be reconstructed exactly from its samples if the highest frequency present in the signal (*f*max) is no greater than half the sampling frequency (*F*s ).
  - **Anti-aliasing** **filter**（抗混叠滤波器）： To suppress the energy at frequencies above the Nyquist frequency from the analog signal being recorded.
  - **Why 44100HZ？**：Humans can hear frequencies between 20 Hz and 20 kHz.

## Chapter 3: Quantization and Dynamic Range

- **Quantization**：
  - **Bits**（位数）：The more bits used to represent an analog value the more precise the representation should be.![](C:\Users\Crazbeans\Desktop\HappyThing\音频处理\image-20250106230409.png)
- **SNR**（信噪比）:
  - It is the **ratio** of the power of the correct signal to the noise added by the quantization process.
  - The SNR depends on the **bit depth** (number of bits) chosen for the signal.
- **Dynamic Range**（动态范围）：
  - It is the **ratio** between the largest amplitude and the smallest amplitude of the audio signal that can be represented with a given bit depth.
- **Loudness**（响度）：
  - It is the average level of an entire recorded waveform, rather than the peak, which measures a single instance in time.

## Chapter 4: Audio File Formats

- **WAV**: 
  - It's a lossless file format, meaning that there is no data loss whatsoever.
  - Also, it is uncompressed, meaning that the data is stored as-is in full original format that doesn't require decoding.
- **AIFF**:
  - Same as WAV but this format is created for Apple computers.
- **MP3**:
  - With compression algorithms that were capable of achieving impressively small file sizes.
  - But this saving of space means some data has to get lost in the process. Usually, high frequencies are the first ones to go.
- **FLAC**:
  - FLAC offers as rich a range of metadata tags in the way.
  - It has smaller size than WAV.

## Chapter 5: Introduction to Frequency Analysis

- **Fourier Theory**:

  - It is possible to form any periodic function *f*(*t*) from the summation

    of a series of sine and cosine terms of increasing frequency.

- **Phase**:

  - A change in phase is equivalent to a time shift in the sine signal.

  $$
  x(t) = sin(wt + φ)
  $$

  - ```java
    double[] Sine(double Amplitude, double fhz, double Fs, double tdur, double phase) {
            int Len = (int) (tdur * Fs);
            double[] signal = new double[Len];
            double pi = Math.PI;
    
            for (int n = 0; n < Len; n++) {
                signal[n] = Amplitude * Math.sin(2 * pi * n * fhz / Fs + phase);
            }
            return signal;
    }
    ```

- **Frequency spectrum**（频谱）:

  - Phase spectrum:

    <img src="C:\Users\Crazbeans\Desktop\HappyThing\音频处理\image-20250107150648281.png" alt="image-20250107150648281" style="zoom:50%;" />

  - Amplitude Spectrum:

    <img src="C:\Users\Crazbeans\Desktop\HappyThing\音频处理\image-20250107150751053.png" alt="image-20250107150751053" style="zoom:50%;" />

- **Discrete Fourier transform**（离散傅里叶变换）:

  - *Ts=1/Fs*
  - For a signal of length N, DFT requires N summations, and each summation involves N multiplications and additions.

- **Why *Component frequency* differs from *bin frequency*?**

  1. The frequency of the wave is not an **integer number of periods**.
  2. Processed signal's **Endpoints are discontinuous**.

- **Windowing**

  - Rectangular / Hanning windowing
  - Window functions smooth the boundaries of the signal, turning it down gradually at the beginning and end of the window, rather than cutting it off abruptly.
  - Computation:![image-20250107161354812](C:\Users\Crazbeans\Desktop\HappyThing\音频处理\image-20250107161354812.png)


- **Zero-padding**:
  - The magnitude output from the FFT is smoother in appearance
- **Computation of the Spectrogram**:
  - Audio in  ->  Buffer into frames  ->  Window  ->  FFT  ->  Magnitude and Power  ->  Depth Map Image
- **Time Resolution / Frequency Resolution**:
  - Short window length gets better Time Resolution.
  - Long window length gets better Frequency Resolution.

## Chapter 6: Speech and Audio Signal Processing

- **Voiced sounds**:

  - Voiced sounds are produced by forcing air through the glottis with the

    tension of the vocal cords adjusted so that they vibrate in a relaxation

    oscillation, thereby producing quasi-periodic pulses of air which excite

    the vocal tract.

  - Short vowel: /i/ /a/ /ʊ/ /ɛ/ /ʌ/

  - Weak vowel: /ə/ /i/ /u/

  - Diphthong: /iə/ /ʊə/ /eɪ/ /əʊ/

  - Long vowel: /i:/ /u:/ /əː/

  - Nasal consonant sound: /m/ /n/ /ŋ/

- **Unvoiced sounds**:

  - Unvoiced sounds are generated by forming a constriction at some point in the vocal tract and forcing air through the constriction at a high enough velocity to produce turbulence.
  - /p/, /t/, /k/, /f/, /θ/, /s/, /ʃ/, /tʃ/, /h/

- **Fricative sounds**:

  - Voiced sounds: /v, ð, z, ʒ/
  - Unvoiced sounds: /f, θ, s, ʃ, h/

- **Plosive consonants**:

  - /p, b, t, d, k, g/

- **Pitch**:

$$
    F_0 = 1 / T_0
$$

​    



# [Numerical Analysis]()

## P1: Polynomial

<img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116154540665-17391728625811.png" alt="image-20250116154540665" style="zoom:33%;" />

## P2: Conversion between binary/decimal

Decimal to Binary:

<img src="C:\Users\Crazbeans\Desktop\HappyThing\Semester 2024-2\数值计算\image-20250116154617410.png" alt="image-20250116154617410" style="zoom: 50%;" /><img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116154626185-17391728625824.png" alt="image-20250116154626185" style="zoom: 50%;" />

Binary to Decimal:

<img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116155318688-17391728625825.png" alt="image-20250116155318688" style="zoom: 33%;" />

## P3: Error

$$
Absolute\ Error: e(x^*)=x^*-x\\
Relative\ Error: e_r(x^*)=\frac{x^*-x}{x^*}
$$

Suppose *x* is rounded to 2*.*34, the error bound of 2.34 is:
$$
|x-2.34|\le0.5\times10^{-2}
$$

## P4: Bisection method

A solution is **correct within** *p* **decimal places** if the error is less than 0.5×10^-p^.

Therefore, if we need *p* decimal places correction, it takes at most *n* bisection steps, where *n* satisfies:
$$
\frac{b_0-a_0}{2^{n+1}}<0.5\times10^{-p}
$$

## P5: Fixed-point iteration

Determine the convergence or divergence for functions:

<img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116164631021-17391728625827.png" alt="image-20250116164631021" style="zoom: 33%;" />

## P6: Newton's method

**Rate of Convergence**: Let x^*^ be a root of *x = F(x)*, and denote e~k~=x~k~-x^*^. If
$$
\lim\limits_{k \to \infty}|\frac{e_{k+1}}{e^{p}_{k}}|=c\ \ \ \ (k\rightarrow\infty,c\ne0)
$$

then we say the iteration form *x~k+1~ = F(x~k~)* has a rate of convergence of *p*.

- when *p* = 1, we call it linear convergence, and it requires *c* *<* 1.



**Newton's method**:
<img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116170507204-17391728625823.png" alt="image-20250116170507204" style="zoom:50%;" />

## P7: LU factorization

<img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116185529322-17391728625826.png" alt="image-20250116185529322" style="zoom:50%;" />

<img src="C:\Users\Crazbeans\Desktop\新建文件夹\image-20250116185732818-17391728625828.png" alt="image-20250116185732818" style="zoom:50%;" />

## P8: Jacobi and the Gauss-Seidel method

 ![image-20250116204031528](C:\Users\Crazbeans\Desktop\HappyThing\数值计算\image-20250116204031528.png)

## P9: Newton interpolating polynomial

![image-20250116223855510](C:\Users\Crazbeans\Desktop\HappyThing\数值计算\image-20250116223855510.png)









# [Artificial Intelligence & NLP]()

## Chapter 1: Neural Networks

### Perceptron & MLP

- **Concepts**:

  - It is a linear binary classifier, typically used for **supervised learning**.

  - A multi-layer perceptron is called **artificial neural networks (ANNs)**.

- **Components**:

  - ![image-20250111171628357](C:\Users\Crazbeans\Desktop\HappyThing\NLP\image-20250111171628357.png)

  - Input values:
    - Directly impact how the perceptron calculates weighted sums and determines the activation.
  - Weights and Bias:
    - Both parameters are adjusted to reach the desired values and output, as the network is trained via **forward propagation and backpropagation.**
    - Weights are used to determine how important each feature is in forecasting output value.
    - Biases allow the model to shift the activation function and help in better fitting the data.
  - Weighted sum:
    - It is continuously updated during training through **backpropagation** to minimize the loss.
  - Activation function

- **Limitations of Perceptron**:

  - The perceptron uses a **linear activation function** so it cannot introduce non-linearity into the learning process (like XOR). 

  - The perceptron outputs **binary decisions (0 or 1) with no probabilistic** 

    **interpretation of the output**. This also means it has no support for **multi**

    **class classification** problems.

  - The perceptron is a **single-layer neural network**, meaning it has only one 

    layer of weights. It relies entirely on the input features provided, without the 

    capability to discover or learn relevant intermediate representations.

- **Non-linear Activation Functions**:

  - Sigmoid: Produces outputs between 0 and 1, useful for probabilistic interpretations.
  - Softmax: Used in multi-class classification problems to output probabilities that sum to 1.

- **Multi-Layer Perceptron (MLP)**:

  - A perceptron with **multiple layers**, **non-linear activation functions**, and various **loss functions** can now solve complex, non-linear problems and handle multi-class classification.

### CNNs

- ![image-20250111220125197](C:\Users\Crazbeans\Desktop\HappyThing\NLP\image-20250111220125197.png)

- **Convolution kernel**（卷积核），**Stride**（步长）

- The **fully connected (FC) layer** (or dense layer) replaces have full connectivity 

  with all neurons in the preceding and succeeding layer as seen in regular MLP.

- Nouns:

  - The **learning rate** controls how big a step the optimizer takes in the 

    gradient direction. A too-high learning rate can prevent convergence, while 

    a too-low learning rate may result in slow training.

  - Choose a **batch** size (number of samples per gradient update). Smaller 

    batches can lead to faster generalization, while larger batches can speed up 

    computation with stable gradients but might overfit more easily.

  - The **activation function** determines the output of individual neurons in a neural network, defining the non-linearity of the network. Without activation functions, the model would just be a linear transformation, which would limit its ability to model complex data.

  - The **loss function** quantifies the difference between the **predicted** **output** of the model and the **actual target** value. During training, the goal is to minimize this loss function, which measures how well the network is performing.

  - The **optimizer** is responsible for adjusting the model's weights during training based on the computed loss, using techniques like **gradient descent**. Its goal is to find the set of weights that minimize the loss function.

  - **Regularization** reduces **overfitting** and improves the **generalization** ability of a model. Overfitting occurs when a model performs exceptionally well on training data but fails to generalize to test data.

### Transfer Learning

- Pros:
  - Reduced Training Time
  - Better Performance
  - Reduced Data Requirement
- Cons:
  - Dissimilar tasks
  - When sufficient data is available

## Chapter 2: Natural Language Processing

### Core Technologies

- **Tokenization:** 
  - Break text into smaller units called "tokens".
- **Sentence Splitting:** 
  - Divide text into sentences for meaningful analysis.
- **Language Model** (LM) :
  - Predicts the next word in a sequence based on previous words to capture the structure and patterns of language to understand context.
- **Part of Speech (POS) Tagging**:
  - Determine each word's role and syntactic due to ambiguity in words.
  - **Techniques**: Logistic Regression, Decision Trees, Random Forests, Support Vector Machines
- **Parsing**:
  - Analyzing a sequence of tokens based on grammar rules to derive sentence structure.
- **Named Entity Recognition**:
  - Identifies and classifies entities in text into predefined categories.
- **Multi-Task Learning (MTL)**:
  - Training a model to perform multiple related tasks simultaneously.
  - Shares knowledge between tasks, improving performance across tasks.
- **Multi-Modal Learning (MML)**:
  - Integrates data from multiple modalities (eg., text, images, audio).
  - Enhances understanding by combining complementary information sources.

### Applications

- **Text Classification (TC)**
- **Machine Translation (MT)**
- **Information Retrieval (IR)**
- **Question Answering (QA)**

### Word Embeddings

**Non-Contextualized Word Embeddings**

- Generate a single vector representation for each word, regardless of its context (**static representation**), such as Word2Vec, GloVe, and FastText.

- **Word2Vec**
  - Create word embeddings using **local co-occurrence** within small context windows in the corpus.
  - **CBOW**: Predict center word from surrounding words. Effective for capturing relationships in smaller datasets.
  - **Skip-Gram**: Predict surrounding words from the center word. Faster to train and is effective on larger datasets.

**Contextualized Word Embeddings**

- Produce different representations for a word based on its surrounding context (**dynamic representation**), such as ELMo and BERT.

## Chapter 3: Recurrent Neural Networks

### RNNs

- **RNNs** introduce a "memory" mechanism to model sequences and capture temporal dependencies.

  > Process one element at a time while maintaining a hidden state to capture context from previous elements.

- **Long Short-Term Memory (LSTM)**:

  - Standard RNNs struggle with **long-term dependencies** due to vanishing gradients.
  - LSTMs are a type of RNN designed to overcome limitations of standard RNNs. They can **remember long-term dependencies** effectively.

### Transformer

- Architecture:                              <img src="C:\Users\Crazbeans\Desktop\HappyThing\NLP\image-20250112163846634.png" alt="image-20250112163846634" style="zoom: 33%;" />

- **Encoder**

  - Processes the input sequence in parallel, applying multiple layers of self-attention and feedforward networks. It generates **contextualized representations** of the input.

- **Decoder**

  - Generates the output sequence step-by-step, using masked self-attention, encoder-decoder attention (to attend to encoder outputs), and feedforward layers to predict the next token.

- **Positional Encoding**

  - Since transformers don't process data sequentially (like RNNs), positional encodings are added to input embeddings to **inject information about the position of tokens in the** **sequence.**

- **Self-Attention & Multi-Head Attention mechanism**

  - Enables the model to focus on different parts of the sequence **simultaneously** and capturing various types of relationships, enhancing context understanding.

- **Residual Connections**

  - Allow the input of each layer to bypass the transformations and be added to the output, helping with gradient flow and **enabling deeper models**.

- **Layer Normalization**

  - Normalizes the activations of each layer, improving **training stability** and helping the model generalize better across different sequences.

- **Advantages**:

  - **Parallelization**
    - Transformers process entire sequences simultaneously, speeding up training and inference compared to RNNs/LSTMs.

  - **Long-Range Dependencies**
    - Self-attention captures relationships across the entire sequence, which captures global context, ideal for tasks like translation.

  - **Scalability**
    - Transformers scale well with data and model size, but computational complexity increases exponentially with sequence length.

### Application

- **BERT**:

  - **Pre-trained** on large corpora.
  - Model learns to predict masked tokens based on context from both directions.
  - Predict if two sentences are sequential in the text.
  - **Fine-tuned** for various NLP tasks.

- **GPT**:

  - **In-Context Learning (ICL)**

    - Introduced the ability to perform tasks in **a few-shot or zero** **shot manner.**

    - Tasks are taught via natural language text, aligning pre-training 

      and usage under the same paradigm.

    - Predicts task solutions as text sequences, given task descriptions 

      and demonstrations (known as prompts).

## Chapter 4: Evolutionary Computation

### Genetic Algorithms

- **Initialization**
  - Start by generating an initial **population** of candidate solutions, often **randomly**. Each candidate often called a "**chromosome**".
- **Fitness Function**
- **Selection Methods**
  - **Raw Fitness Selection**
    - **Simple** and **intuitive**, but **low diversity**.
  - **Roulette Wheel Selection**
    - **Fair** chance, but **early convergence**.
  - **Tournament Selection**
    - **Good diversity**, but may **miss best individuals** if N is small.
- **Crossover**
  - **Single-Point Crossover**
  - **Two-Point Crossover**
  - **Uniform Crossover**
- **Mutation**
  - Mutation introduces **random changes** to individuals to maintain genetic diversity and avoid **early convergence**.

### GA with MLP

- Differences:

  - **Genes**: Represent the weights and biases of the neural network. 

    These are usually flattened into a single vector.

  - **Chromosome**: A complete set of weights and biases for the entire 

    network.

- **Pros**

  - GAs don't rely on gradients, so they can optimize non-differentiable or discontinuous loss functions.
  - GAs evaluate multiple solutions simultaneously, making them suitable for parallel computation.

- **Cons**:

  - Performance of GAs depends on hyperparameters like population size, mutation rate, and crossover rate, which can be hard to tune.
  - GAs might not fine-tune weights as precisely as backpropagation due to the random of the search process.







































































