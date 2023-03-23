# BERTAgent

## Quick Start

Here we provide quick and easy setup and example usage for BERTAgent. The code provided herein requires CUDA. If CUDA is not available locally an online service can be used such as Google Colab (https://colab.research.google.com/) or Kaggle (https://www.kaggle.com/). For example, https://colab.research.google.com/drive/19fdmML86orvH0AAfV6CQ4EVW6aI-DNcG?usp=sharing

### Assert that CUDA is available


```python
import torch

# Assert CUDA is currently available.
assert torch.cuda.is_available(), "No CUDA!"

# Alternatively use the following commands to check CUDA status:
#!nvidia-smi
#!nvidia-smi --query-gpu=name --format=csv,noheader
```

### Install BERTAgent

NB: The code below needs to be run only once per environment/runtime instance.


```python
# Uncomment the line below to install BERTAgent
# and its dependencies (transformers, pandas, etc).
!pip install --upgrade --pre bertagent
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: bertagent in /usr/local/lib/python3.8/dist-packages (0.0.10)
    Requirement already satisfied: datasets>=2.1.0 in /usr/local/lib/python3.8/dist-packages (from bertagent) (2.10.1)
    Requirement already satisfied: huggingface-hub>=0.5.1 in /usr/local/lib/python3.8/dist-packages (from bertagent) (0.12.1)
    Requirement already satisfied: pandas>=1.4.2 in /usr/local/lib/python3.8/dist-packages (from bertagent) (2.0.0rc0)
    Requirement already satisfied: transformers>=4.17.0 in /usr/local/lib/python3.8/dist-packages (from bertagent) (4.26.1)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (4.64.1)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (2023.1.0)
    Requirement already satisfied: responses<0.19 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (0.18.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (23.0)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (2.25.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (0.70.14)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (3.8.4)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (3.2.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (6.0)
    Requirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (9.0.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (1.22.4)
    Requirement already satisfied: dill<0.3.7,>=0.3.0 in /usr/local/lib/python3.8/dist-packages (from datasets>=2.1.0->bertagent) (0.3.6)
    Requirement already satisfied: filelock in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.5.1->bertagent) (3.9.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.5.1->bertagent) (4.5.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.8/dist-packages (from pandas>=1.4.2->bertagent) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.8/dist-packages (from pandas>=1.4.2->bertagent) (2022.7.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.8/dist-packages (from transformers>=4.17.0->bertagent) (2022.6.2)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.8/dist-packages (from transformers>=4.17.0->bertagent) (0.13.2)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (3.0.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (6.0.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (1.8.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (22.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (1.3.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets>=2.1.0->bertagent) (1.3.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.8.2->pandas>=1.4.2->bertagent) (1.15.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets>=2.1.0->bertagent) (2022.12.7)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets>=2.1.0->bertagent) (2.10)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets>=2.1.0->bertagent) (1.26.14)
    Requirement already satisfied: chardet<5,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets>=2.1.0->bertagent) (4.0.0)


### Basic Imports


```python
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import bertagent
from bertagent import SentencesPredictor
from bertagent import df0
print(bertagent.__version__)
```

    0.0.10


### Initialize predictor

NB: If needed the BERTAgent model will be downloaded (this should happen only one per environment/runtime.


```python
predictor = SentencesPredictor()
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/813 [00:00<?, ?B/s]



    Downloading (…)"pytorch_model.bin";:   0%|          | 0.00/438M [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]


### Predict Linguistic agency on sample data


```python
df0 = df0.copy()
df0["predict"] = df0.sents.progress_apply(predictor.predict)
df0
```

    100%|██████████| 43/43 [00:04<00:00,  9.57it/s]






  <div id="df-6647d6d3-e7b1-45ba-ab71-7327b86e2e1e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sents</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[first sentence, we must adhere, we must strive]</td>
      <td>[0.4445924758911133, 1.1372489929199219, 2.482...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[dummy test, another stupid idea, she did a br...</td>
      <td>[-0.7290191650390625, -2.3433375358581543, 2.1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[just one sentence here]</td>
      <td>[-0.08561539649963379]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[This is a book.]</td>
      <td>[0.2554454803466797]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[I've been waiting for this amazing thing my w...</td>
      <td>[0.7085838317871094]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[I am fed-up with waiting.]</td>
      <td>[0.13408374786376953]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[They are ridiculous]</td>
      <td>[-1.7097516059875488]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[They are amazing]</td>
      <td>[0.9146566390991211]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[We are winners]</td>
      <td>[2.0414867401123047]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[We are losers]</td>
      <td>[-1.517617106437683]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[she is a hard working individual]</td>
      <td>[1.7002677917480469]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[she is a hardly working individual]</td>
      <td>[-1.2014453411102295]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[We are motivated]</td>
      <td>[2.638154983520508]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[We are not so motivated]</td>
      <td>[-1.512332558631897]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[We must win]</td>
      <td>[2.138918876647949]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[We'll lose]</td>
      <td>[-1.3526885509490967]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[Strive to achieve some goal]</td>
      <td>[2.72894287109375]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[Lazy and unmotivated]</td>
      <td>[-2.7035977840423584]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[well planned and well executed]</td>
      <td>[1.9415154457092285]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[Everything is messy and uncoordinated]</td>
      <td>[-0.3087284564971924]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[uncoordinated activity]</td>
      <td>[0.3766012191772461]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[coordinated activity]</td>
      <td>[1.5255565643310547]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[unpredictible decisionmaker]</td>
      <td>[1.6449546813964844]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[bad decisionmaker]</td>
      <td>[-1.1596208810806274]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[marvelous decisionmaker]</td>
      <td>[2.1274008750915527]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[anti logic]</td>
      <td>[0.7003819942474365]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>[anti-logic]</td>
      <td>[0.7914280891418457]</td>
    </tr>
    <tr>
      <th>27</th>
      <td>[i am very dissapointed with this decission]</td>
      <td>[0.1831493377685547]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[we must fight for our rights]</td>
      <td>[2.196908950805664]</td>
    </tr>
    <tr>
      <th>29</th>
      <td>[this is a car it runs on gas]</td>
      <td>[0.06075859069824219]</td>
    </tr>
    <tr>
      <th>30</th>
      <td>[I am afraid that I am sure that we will win]</td>
      <td>[-0.025776386260986328]</td>
    </tr>
    <tr>
      <th>31</th>
      <td>[I am afraid that I am sure that we will lose]</td>
      <td>[-1.0667026042938232]</td>
    </tr>
    <tr>
      <th>32</th>
      <td>[]</td>
      <td>[0.43352603912353516]</td>
    </tr>
    <tr>
      <th>33</th>
      <td>[yes]</td>
      <td>[0.7879881858825684]</td>
    </tr>
    <tr>
      <th>34</th>
      <td>[no]</td>
      <td>[-0.5536911487579346]</td>
    </tr>
    <tr>
      <th>35</th>
      <td>[maybe]</td>
      <td>[-0.22850513458251953]</td>
    </tr>
    <tr>
      <th>36</th>
      <td>[He who would live must fight.]</td>
      <td>[1.5349712371826172]</td>
    </tr>
    <tr>
      <th>37</th>
      <td>[He who doesn’t wish to fight in this world, w...</td>
      <td>[-0.5808243751525879]</td>
    </tr>
    <tr>
      <th>38</th>
      <td>[I know that fewer people are won over by the ...</td>
      <td>[0.19835186004638672]</td>
    </tr>
    <tr>
      <th>39</th>
      <td>[that every great movement on this earth owes ...</td>
      <td>[-0.010506153106689453]</td>
    </tr>
    <tr>
      <th>40</th>
      <td>[We should take control and assert our position]</td>
      <td>[2.460244655609131]</td>
    </tr>
    <tr>
      <th>41</th>
      <td>[We should give up and say nothing]</td>
      <td>[-1.1041367053985596]</td>
    </tr>
    <tr>
      <th>42</th>
      <td>[no way]</td>
      <td>[-0.41170215606689453]</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6647d6d3-e7b1-45ba-ab71-7327b86e2e1e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6647d6d3-e7b1-45ba-ab71-7327b86e2e1e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6647d6d3-e7b1-45ba-ab71-7327b86e2e1e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Produce separate columns for BERTAgent whole, positive, negative and absolute aggregation of results


```python
df0["BA_whole"] = df0.predict.apply(predictor.BA_whole)
df0["BA_posit"] = df0.predict.apply(predictor.BA_posit)
df0["BA_negat"] = df0.predict.apply(predictor.BA_negat)
df0["BA_absol"] = df0.predict.apply(predictor.BA_absol)
df0
```





  <div id="df-d95d4a38-dbc7-4ba5-be39-3b0dda4ab539">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sents</th>
      <th>predict</th>
      <th>BA_whole</th>
      <th>BA_posit</th>
      <th>BA_negat</th>
      <th>BA_absol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[first sentence, we must adhere, we must strive]</td>
      <td>[0.4445924758911133, 1.1372489929199219, 2.482...</td>
      <td>1.354729</td>
      <td>1.354729</td>
      <td>0.000000</td>
      <td>1.354729</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[dummy test, another stupid idea, she did a br...</td>
      <td>[-0.7290191650390625, -2.3433375358581543, 2.1...</td>
      <td>-0.301695</td>
      <td>0.722424</td>
      <td>1.024119</td>
      <td>1.746543</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[just one sentence here]</td>
      <td>[-0.08561539649963379]</td>
      <td>-0.085615</td>
      <td>0.000000</td>
      <td>0.085615</td>
      <td>0.085615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[This is a book.]</td>
      <td>[0.2554454803466797]</td>
      <td>0.255445</td>
      <td>0.255445</td>
      <td>0.000000</td>
      <td>0.255445</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[I've been waiting for this amazing thing my w...</td>
      <td>[0.7085838317871094]</td>
      <td>0.708584</td>
      <td>0.708584</td>
      <td>0.000000</td>
      <td>0.708584</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[I am fed-up with waiting.]</td>
      <td>[0.13408374786376953]</td>
      <td>0.134084</td>
      <td>0.134084</td>
      <td>0.000000</td>
      <td>0.134084</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[They are ridiculous]</td>
      <td>[-1.7097516059875488]</td>
      <td>-1.709752</td>
      <td>0.000000</td>
      <td>1.709752</td>
      <td>1.709752</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[They are amazing]</td>
      <td>[0.9146566390991211]</td>
      <td>0.914657</td>
      <td>0.914657</td>
      <td>0.000000</td>
      <td>0.914657</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[We are winners]</td>
      <td>[2.0414867401123047]</td>
      <td>2.041487</td>
      <td>2.041487</td>
      <td>0.000000</td>
      <td>2.041487</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[We are losers]</td>
      <td>[-1.517617106437683]</td>
      <td>-1.517617</td>
      <td>0.000000</td>
      <td>1.517617</td>
      <td>1.517617</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[she is a hard working individual]</td>
      <td>[1.7002677917480469]</td>
      <td>1.700268</td>
      <td>1.700268</td>
      <td>0.000000</td>
      <td>1.700268</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[she is a hardly working individual]</td>
      <td>[-1.2014453411102295]</td>
      <td>-1.201445</td>
      <td>0.000000</td>
      <td>1.201445</td>
      <td>1.201445</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[We are motivated]</td>
      <td>[2.638154983520508]</td>
      <td>2.638155</td>
      <td>2.638155</td>
      <td>0.000000</td>
      <td>2.638155</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[We are not so motivated]</td>
      <td>[-1.512332558631897]</td>
      <td>-1.512333</td>
      <td>0.000000</td>
      <td>1.512333</td>
      <td>1.512333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[We must win]</td>
      <td>[2.138918876647949]</td>
      <td>2.138919</td>
      <td>2.138919</td>
      <td>0.000000</td>
      <td>2.138919</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[We'll lose]</td>
      <td>[-1.3526885509490967]</td>
      <td>-1.352689</td>
      <td>0.000000</td>
      <td>1.352689</td>
      <td>1.352689</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[Strive to achieve some goal]</td>
      <td>[2.72894287109375]</td>
      <td>2.728943</td>
      <td>2.728943</td>
      <td>0.000000</td>
      <td>2.728943</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[Lazy and unmotivated]</td>
      <td>[-2.7035977840423584]</td>
      <td>-2.703598</td>
      <td>0.000000</td>
      <td>2.703598</td>
      <td>2.703598</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[well planned and well executed]</td>
      <td>[1.9415154457092285]</td>
      <td>1.941515</td>
      <td>1.941515</td>
      <td>0.000000</td>
      <td>1.941515</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[Everything is messy and uncoordinated]</td>
      <td>[-0.3087284564971924]</td>
      <td>-0.308728</td>
      <td>0.000000</td>
      <td>0.308728</td>
      <td>0.308728</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[uncoordinated activity]</td>
      <td>[0.3766012191772461]</td>
      <td>0.376601</td>
      <td>0.376601</td>
      <td>0.000000</td>
      <td>0.376601</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[coordinated activity]</td>
      <td>[1.5255565643310547]</td>
      <td>1.525557</td>
      <td>1.525557</td>
      <td>0.000000</td>
      <td>1.525557</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[unpredictible decisionmaker]</td>
      <td>[1.6449546813964844]</td>
      <td>1.644955</td>
      <td>1.644955</td>
      <td>0.000000</td>
      <td>1.644955</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[bad decisionmaker]</td>
      <td>[-1.1596208810806274]</td>
      <td>-1.159621</td>
      <td>0.000000</td>
      <td>1.159621</td>
      <td>1.159621</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[marvelous decisionmaker]</td>
      <td>[2.1274008750915527]</td>
      <td>2.127401</td>
      <td>2.127401</td>
      <td>0.000000</td>
      <td>2.127401</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[anti logic]</td>
      <td>[0.7003819942474365]</td>
      <td>0.700382</td>
      <td>0.700382</td>
      <td>0.000000</td>
      <td>0.700382</td>
    </tr>
    <tr>
      <th>26</th>
      <td>[anti-logic]</td>
      <td>[0.7914280891418457]</td>
      <td>0.791428</td>
      <td>0.791428</td>
      <td>0.000000</td>
      <td>0.791428</td>
    </tr>
    <tr>
      <th>27</th>
      <td>[i am very dissapointed with this decission]</td>
      <td>[0.1831493377685547]</td>
      <td>0.183149</td>
      <td>0.183149</td>
      <td>0.000000</td>
      <td>0.183149</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[we must fight for our rights]</td>
      <td>[2.196908950805664]</td>
      <td>2.196909</td>
      <td>2.196909</td>
      <td>0.000000</td>
      <td>2.196909</td>
    </tr>
    <tr>
      <th>29</th>
      <td>[this is a car it runs on gas]</td>
      <td>[0.06075859069824219]</td>
      <td>0.060759</td>
      <td>0.060759</td>
      <td>0.000000</td>
      <td>0.060759</td>
    </tr>
    <tr>
      <th>30</th>
      <td>[I am afraid that I am sure that we will win]</td>
      <td>[-0.025776386260986328]</td>
      <td>-0.025776</td>
      <td>0.000000</td>
      <td>0.025776</td>
      <td>0.025776</td>
    </tr>
    <tr>
      <th>31</th>
      <td>[I am afraid that I am sure that we will lose]</td>
      <td>[-1.0667026042938232]</td>
      <td>-1.066703</td>
      <td>0.000000</td>
      <td>1.066703</td>
      <td>1.066703</td>
    </tr>
    <tr>
      <th>32</th>
      <td>[]</td>
      <td>[0.43352603912353516]</td>
      <td>0.433526</td>
      <td>0.433526</td>
      <td>0.000000</td>
      <td>0.433526</td>
    </tr>
    <tr>
      <th>33</th>
      <td>[yes]</td>
      <td>[0.7879881858825684]</td>
      <td>0.787988</td>
      <td>0.787988</td>
      <td>0.000000</td>
      <td>0.787988</td>
    </tr>
    <tr>
      <th>34</th>
      <td>[no]</td>
      <td>[-0.5536911487579346]</td>
      <td>-0.553691</td>
      <td>0.000000</td>
      <td>0.553691</td>
      <td>0.553691</td>
    </tr>
    <tr>
      <th>35</th>
      <td>[maybe]</td>
      <td>[-0.22850513458251953]</td>
      <td>-0.228505</td>
      <td>0.000000</td>
      <td>0.228505</td>
      <td>0.228505</td>
    </tr>
    <tr>
      <th>36</th>
      <td>[He who would live must fight.]</td>
      <td>[1.5349712371826172]</td>
      <td>1.534971</td>
      <td>1.534971</td>
      <td>0.000000</td>
      <td>1.534971</td>
    </tr>
    <tr>
      <th>37</th>
      <td>[He who doesn’t wish to fight in this world, w...</td>
      <td>[-0.5808243751525879]</td>
      <td>-0.580824</td>
      <td>0.000000</td>
      <td>0.580824</td>
      <td>0.580824</td>
    </tr>
    <tr>
      <th>38</th>
      <td>[I know that fewer people are won over by the ...</td>
      <td>[0.19835186004638672]</td>
      <td>0.198352</td>
      <td>0.198352</td>
      <td>0.000000</td>
      <td>0.198352</td>
    </tr>
    <tr>
      <th>39</th>
      <td>[that every great movement on this earth owes ...</td>
      <td>[-0.010506153106689453]</td>
      <td>-0.010506</td>
      <td>0.000000</td>
      <td>0.010506</td>
      <td>0.010506</td>
    </tr>
    <tr>
      <th>40</th>
      <td>[We should take control and assert our position]</td>
      <td>[2.460244655609131]</td>
      <td>2.460245</td>
      <td>2.460245</td>
      <td>0.000000</td>
      <td>2.460245</td>
    </tr>
    <tr>
      <th>41</th>
      <td>[We should give up and say nothing]</td>
      <td>[-1.1041367053985596]</td>
      <td>-1.104137</td>
      <td>0.000000</td>
      <td>1.104137</td>
      <td>1.104137</td>
    </tr>
    <tr>
      <th>42</th>
      <td>[no way]</td>
      <td>[-0.41170215606689453]</td>
      <td>-0.411702</td>
      <td>0.000000</td>
      <td>0.411702</td>
      <td>0.411702</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d95d4a38-dbc7-4ba5-be39-3b0dda4ab539')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d95d4a38-dbc7-4ba5-be39-3b0dda4ab539 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d95d4a38-dbc7-4ba5-be39-3b0dda4ab539');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
predictor.predict(["made of lead"])
```




    [0.1323385238647461]




```python
predictor.predict(["we will lead them"])
```




    [1.345942497253418]




```python
predictor.predict(["we will not lead them"])
```




    [0.15259265899658203]




```python
predictor.predict(["we are unable to lead them"])
```




    [-2.10258150100708]




```python
print(f"{predictor.predict(['the man killed a pig'])}")
print(f"{predictor.predict(['the pig was killed by a man'])}")

print(f"{predictor.predict(['strive to survive'])}")
print(f"{predictor.predict(['strive to kill'])}")

```

    [0.21040630340576172]
    [-0.5099365711212158]
    [2.4385910034179688]
    [2.1709980964660645]



```python
print(f"{predictor.predict(['the man killed other people'])}")

```

    [0.8408107757568359]



```python
print(f"{predictor.predict(['the man killed other people'])}")


```

    [0.8408098220825195, 0.8714971542358398]



```python

```
