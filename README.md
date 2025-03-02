<div align="center">

# C-3PO: Compact Plug-and-Play Proxy Optimization to Achieve Human-like Retrieval-Augmented Generation

<a href='https://arxiv.org/abs/2502.06205'><img src='https://img.shields.io/badge/Paper-Arxiv-red'> </a>
<a href='https://chen-gx.github.io/projects/C-3PO/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>

This repo contains a proxy-centric alignment framework (C-3PO) that bridges the gap between retrievers and LLMs. Instead of modifying existing components in RAG, C-3PO introduces multi-agent system within a lightweight proxy model to simulate humen-like behaviors that optimizes the entire RAG pipeline while maintaining plug-and-play compatibility.

<div align="center">
<img src="./images/C-3PO.png" width="90%" alt="c-3po_framework">
</div>

## :boom: News
- **[2025.03.02]** Release our Code.
- **[2025.02.12]** Release our [Demo](https://www.modelscope.cn/studios/Decaderan/C-3PO) on the ModelScope.
- **[2025.02.10]** Release our paper [C-3PO](https://arxiv.org/abs/2502.06205) on the Arxiv.

<!-- # :pushpin: TODO 

The code and interactive demo are currently under preparation and will be available soon. -->


## :honeybee: Deploy LLM and Retrieval services

### Step1: Python Environment
For C-3PO (also works for LLM server)
```bash
conda create -n c3po python=3.11
conda activate c3po
pip install -r requirements.txt
```

For Retrieval (dense model)
```bash
conda create -n faiss python=3.11
conda activate faiss
pip install -r retrieval_requirements.txt
```

### Step2: Download LLM from Huggingface
Please download the following models from huggingface:
```
Qwen2-0.5B
Qwen2-1.5B
Qwen2-72B-Instruct
contriever-msmarco
```

### Step3: Download the wikipedia 2018 dump
Download preprocessed passage data of the wikipedia 2018 dump.
```bash
cd ./C-3PO/deploy_servers/retrieve_server/wiki18
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```
Then, download the embedded passages. We use Contriever-MSMARCO.
```bash
cd ./C-3PO/deploy_servers/retrieve_server/wikipedia_embeddings
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```

### Step4: Deploy the LLM server

```bash
cd ./deploy_servers/llm_server
bash qwen_72b_serve.sh
```

### Step5: Deploy the Retrieval server

```bash
cd ./C-3PO/deploy_servers/retrieve_server/retrieve_code/wiki18
bash start_wiki18.sh
```

## :dart: Inference

### Download our released ckpt (Optional)
You can download our ckpt of [C-3PO-1.5B](https://www.modelscope.cn/models/Decaderan/C-3PO-1.5B) or [C-3PO-0.5B](https://www.modelscope.cn/models/Decaderan/C-3PO-0.5B) on the ModelScope.


### Scripts
Our implementation supports two high-performance inference engines: SGLang and vLLM, allowing users to optimize for different deployment scenarios and hardware configurations.
```bash
cd ./C-3PO/inference
bash single_model.sh
```

## Tree-structured Rollout for Seed Data (Supervised Warm-up)
We collect seed data through tree-structured rollout using Qwen-2-72B-Instruct.

### Step1: tree-structured rollout
```bash
cd ./C-3PO/instruct_sampling_scripts
bash run_72b.sh
```

### Step2: supervised fine-tuning
We use Llama-Factory as our training framework for sft.
```bash
git clone (from llama-Factory)
# we release our training hyper-parameters for easy reproduction.
cd ./C-3PO/train/sft_scripts
bash run_base_packing.sh
```

## :heart: Acknowledgements

This work is built upon several excellent open-source projects. We sincerely thank:

- [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) for providing the supervised fine-tuning framework
- [vLLM](https://github.com/vllm-project/vllm) for the efficient inference engine with high throughput
- [SGLang](https://github.com/sgl-project/sglang) for the efficient inference engine with high throughput
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for the comprehensive reinforcement learning framework

We express our gratitude to all these projects for their outstanding contributions to the open-source community.


## Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{chen2025c,
  title={C-3PO: Compact Plug-and-Play Proxy Optimization to Achieve Human-like Retrieval-Augmented Generation},
  author={Chen, Guoxin and Liao, Minpeng and Yu, Peiying and Wang, Dingmin and Qiao, Zile and Yang, Chao and Zhao, Xin and Fan, Kai},
  journal={arXiv preprint arXiv:2502.06205},
  year={2025}
}
```
Your support by starring ⭐ this repository would be greatly appreciated!