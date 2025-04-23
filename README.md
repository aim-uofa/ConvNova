<p align="center"> <img src="logo.jpg" alt="ConvNova" width="220"/> </p>
<h1>[ICLR2025] ConvNova 🧬 Revisiting Convolution Architecture in the Realm of DNA Foundation Models</h1>

<p>
  <a href="https://openreview.net/forum?id=B07dLVWLyD">OpenReview</a> | 
  <a href="https://arxiv.org/abs/2502.18538">arXiv</a> | 
  <a href="https://github.com/aim-uofa/ConvNova">GitHub</a> | 
  <a href="">HuggingFace 🤗(coming soon)</a>
<!--   <a href="https://huggingface.co/collections/convnova">HuggingFace 🤗</a> <!-- collection will be public on release day --> -->
</p>

<p>
  ConvNova re-examines classical CNNs for genomic language modelling and shows that <strong>dilated convolutions, gated convolutions, and a dual-branch gating design</strong> allow a <em>7 M-parameter</em> network to match or exceed SSM- and Transformer-based backbones such as HyenaDNA and Caduceus on <strong>Genomic Benchmarks, Nucleotide Transformer tasks, and the Long-Range Benchmark</strong> while training and running faster.
</p>

---

<h2>1 Using ConvNova with 🤗 Transformers</h2>
<p>Pre-trained checkpoints (131 k tokens) are released on the HuggingFace Hub:</p>
<table>
  <tr>
    <th>Model</th>
    <th>d_model</th>
    <th>Layers</th>
    <th>Steps</th>
    <th>RC handling</th>
  </tr>
  <tr>
    <td><strong>ConvNova-Ph</strong></td>
    <td>256</td>
    <td>16</td>
    <td>50 k</td>
    <td>RC data-aug</td>
  </tr>
  <tr>
    <td><strong>ConvNova-PS</strong></td>
    <td>256</td>
    <td>16</td>
    <td>50 k</td>
    <td>RC-equivariant</td>
  </tr>
</table>

<pre>
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "convnova/convnova-ph_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForMaskedLM.from_pretrained(model_id)
</pre>

<p>To initialise a smaller scratch model:</p>

<pre>
from transformers import AutoConfig, AutoModelForMaskedLM
config = AutoConfig.from_pretrained(
    model_id, d_model=128, n_layer=8
)
model  = AutoModelForMaskedLM.from_config(config)
</pre>

<p><em>ConvNova uses the standard `K-merTokenizer` shipped with HyenaDNA</em> .</p>

---

<h2>2 Quick start</h2>

<pre>
conda env create -f convnova_env.yml
conda activate convnova_env

# suggested folders
mkdir -p data/hg38 outputs logs
</pre>

<p>The <strong>environment YAML</strong> mirrors HyenaDNA & Caduceus dependencies and pins <code>pytorch>=2.2</code>, <code>flash-attention-cuda20x</code>, <code>transformers>=4.40</code>, and <code>genomic-benchmarks</code>. </p>

---

<h2>3 Reproducing the paper</h2>

<h3>3.1 Pre-training on the Human Reference Genome</h3>
<p>Download the FASTA & BED splits (courtesy of HyenaDNA): </p>

<pre>
mkdir -p data/hg38
curl -L https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz \
   | gunzip -c > data/hg38/hg38.ml.fa
curl -L https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed \
   -o data/hg38/human-sequences.bed
</pre>

<p>Single-node example (4×A100 80 GB):</p>

<pre>
python -m train \
  experiment=hg38/hg38 \
  dataset.max_length=1024 dataset.batch_size=1024 \
  model=convnova model.config.d_model=128 model.config.n_layer=8 \
  optimizer.lr=8e-3 trainer.max_steps=10000 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  wandb=null
</pre>

<p>For Slurm clusters use <code>slurm_scripts/run_pretrain_convnova.sh</code>.</p>

<h3>3.2 Genomic Benchmarks (short-range)</h3>
<p>GenomicBenchmarks provides 8 binary- and multi-class tasks packaged as a Python library. </p>

<pre>
python -m train \
  experiment=hg38/genomic_benchmark \
  dataset.dataset_name="dummy_mouse_enhancers_ensembl" \
  dataset.batch_size=256 \
  model=convnova \
  train.pretrained_model_path="<ckpt>" \
  trainer.max_epochs=10 wandb=null
</pre>

<h3>3.3 Nucleotide Transformer Benchmark</h3>
<p>Datasets are hosted on the Hub as <code>InstaDeepAI/nucleotide_transformer_downstream_tasks</code>. </p>
<p>Use <code>slurm_scripts/run_nucleotide_transformer.sh</code> to sweep all 18 tasks.</p>

<h3>3.4 Long-Range Benchmark (eQTL VEP)</h3>
<p>The <strong>genomics-long-range-benchmark</strong> dataset supplies variant-effect prediction sequences up to 131 k. </p>

<ol>
  <li><strong>Dump embeddings</strong>:
    <pre>
    torchrun --nproc-per-node=8 vep_embeddings.py \
      --seq_len=131072 --embed_dump_batch_size=1 \
      --model_name_or_path convnova/convnova-ps_seqlen-131k_d_model-256_n_layer-16 \
      --rcps
    </pre>
  </li>
  <li><strong>Fit SVM head</strong> in <code>notebooks/vep_svm.ipynb</code>.</li>
</ol>

<h3>3.5 Ablations</h3>
<p>Scripts under <code>ablation/</code> reproduce Table 4 of the paper, contrasting <strong>dilation vs down-sampling</strong> and different gating variants. </p>

---

<h2>4 Citation</h2>

<pre>
@inproceedings{bo2025convnova,
  title     = {Revisiting Convolution Architecture in the Realm of DNA Foundation Models},
  author    = {Yu Bo and Weian Mao and Yanjun Shao and Weiqiang Bai and Peng Ye
               and Xinzhu Ma and Junbo Zhao and Hao Chen and Chunhua Shen},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
</pre>

---

<h2>5 Acknowledgements</h2>
<p>ConvNova builds on the training, logging and data-loading scaffolds of <strong>HyenaDNA</strong> and <strong>Caduceus</strong>, and evaluates on <strong>Genomic Benchmarks</strong>, <strong>Nucleotide Transformer tasks</strong>, and the <strong>Long-Range Benchmark</strong>. We thank the maintainers of these open resources for making rigorous comparison possible. </p>
