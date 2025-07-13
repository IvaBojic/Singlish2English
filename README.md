# Singlish2English
Transcribing and translating Singlish audio to English text.

This repo consists of three parts:
- Code for *fine-tuning* and *inference* of Automatic Speech Recognition (ASR) models on Singapore's National Speech Corpus (NSC) dataset[^1].
- Code for *fine-tuning* and *inference* of Large Language Models (LLMs) on Singlish Health Coaching dataset.
- Code for *end-to-end Singlish-to-English conversion*.


## Fine-tuning and evaluation of ASR models

### Datasets

To enable fine-tuning of open-source foundation ASR models, we curated four bespoke datasets constructed from the NSC corpus, which was manually transcribed with orthographic precision:

- **NSC<sub>P12</sub>**. This dataset consists of non-conversational speech, incorporating recordings of phonetically-balanced scripts with standard English sentences spoken in local accents (Part1 of NSC) alongside recordings of sentences randomly generated from topics such as people, food, locations, and brands (Part2 of NSC).
- **NSC<sub>P35</sub>**. This dataset includes conversational speech, encompassing dialogues on everyday topics (Part3 of NSC) as well as recordings where speakers adopt specific speaking styles, such as debates, finance-related discussions, and expressions of positive and negative emotions (Part5 of NSC).
- **NSC<sub>P15</sub>**. This dataset represents a balanced combination of Part1, Part2, Part3, and Part5 from NSC. 
- **NSC<sub>P16</sub>**. This dataset extends the previous subsets by incorporating recordings of speakers engaged in scenario-based dialogues across various themes (Part6 of NSC).


### ASR models

To determine the optimal fine-tuned ASR model, we fine-tuned four open-source pre-trained models:
- Whisper-small[^2]
- Whisper-medium[^2]
- SeamlessM4T-medium[^3]
- SpeechT5[^4].

Whisper-small and Whisper-medium were each fine-tuned on four manually orographically transcribed datasets: NSC<sub>P12_train</sub>, NSC<sub>P35_train</sub>, NSC<sub>P15_train</sub>, and NSC<sub>P16_train</sub>, resulting in a total of eight fine-tuned models. In contrast, SpeechT5 and SeamlessM4T were fine-tuned on only three datasets each, excluding NSC<sub>P16_train</sub>, to conserve computational resources. This decision was based on their consistently lower performance compared to the Whisper models when fine-tuned on smaller datasets.


### Results

Normalized sentence-level WERs of different fine-tuned ASR models on the NSC<sub>P16_test</sub> set. Columns represent the training set. A lower WER indicates better performance (↓). The **best WER is bolded**, while the _second best is underlined_.

| Model name              | NSC<sub>P12</sub> | NSC<sub>P35</sub> | NSC<sub>P15</sub> | NSC<sub>P16</sub> |
|--------------------------|-------------------|-------------------|-------------------|-------------------|
| Whisper-small            | 12.99             | 8.62              | 7.62              | _7.13_            |
| Whisper-medium           | 11.83             | 7.44              | 6.92              | **6.61**          |
| SeamlessM4T-medium       | 16.83             | 12.66             | 13.03             | -                 |
| SpeechT5                 | 24.33             | 13.15             | 12.17             | -                 |


[^1]: J. X. Koh, A. Mislan, K. Khoo, B. Ang, W. Ang, C. Ng, and Y. Tan, “Building the singapore english national speech corpus,” Malay, vol. 20, no. 25.0, pp. 19–3, 2019
[^2]: OpenAI Whisper: https://github.com/openai/whisper  
[^3]: SeamlessM4T: https://arxiv.org/abs/2308.16824  
[^4]: SpeechT5: https://arxiv.org/abs/2210.14640