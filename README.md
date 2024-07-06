# Pretraining and Finetuning LLMs from the Ground Up



## Overview

This tutorial is aimed at coders interested in understanding the building blocks of large language models (LLMs), how LLMs work, and how to code them from the ground up in PyTorch. We will kick off this tutorial with an introduction to LLMs, recent milestones, and their use cases. Then, we will code a small GPT-like LLM, including its data input pipeline, core architecture components, and pretraining code ourselves. After understanding how everything fits together and how to pretrain an LLM, we will learn how to load pretrained weights and finetune LLMs using open-source libraries.


**The code material is based on my [Build a Large Language Model From Scratch](http://mng.bz/orYv) book and also uses the [LitGPT](https://github.com/Lightning-AI/litgpt) library.**

<br>

## Setup instructions

A ready-to-go cloud environment, complete with all code examples and dependencies installed, is available [here](https://lightning.ai/lightning-ai/studios/llms-from-the-ground-up-workshop?section=recent&view=public). This enables participants to run all code, particularly in the pretraining and finetuning sections, on a GPU.


<div align="center">
<br>


<a target="_blank" href="https://lightning.ai/lightning-ai/studios/llms-from-the-ground-up-workshop">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

<br>
<br>
<br>

</div>

In addition, see the instructions in the [setup](./setup) folder to set up your computer to run the code locally.

## Outline

|      | Title                        | Description                                                  | Folder                               |
| ---- | ---------------------------- | ------------------------------------------------------------ | ------------------------------------ |
| 1    | Introduction to LLMs         | An introduction to the workshop introducing LLMs, the topics being covered in this workshop, and setup instructions. | [01_intro](01_intro)                 |
| 2    | Understanding LLM Input Data | In this section, we are coding the text input pipeline by implementing a text tokenizer and a custom PyTorch DataLoader for our LLM | [02_data](02_data)                   |
| 3    | Coding an LLM architecture   | In this section, we will go over the individual building blocks of LLMs and assemble them in code. We will not cover all modules in meticulous detail but will focus on the bigger picture and how to assemble them into a GPT-like model. | [03_architecture](03_architecture)   |
| 4    | Pretraining LLMs             | In part 4, we will cover the pretraining process of LLMs and implement the code to pretrain the model architecture we implemented previously. Since pretraining is expensive, we will only pretrain it on a small text sample available in the public domain so that the LLM is capable of generating some basic sentences. | [04_pretraining](04_pretraining)     |
| 5    | Loading pretrained weights   | Since pretraining is a long and expensive process, we will now load pretrained weights into our self-implemented architecture. Then, we will introduce the LitGPT open-source library, which provides more sophisticated (but still readable) code for training and finetuning LLMs. We will learn how to load weights of pretrained LLMs (Llama, Phi, Gemma, Mistral) in LitGPT. | [05_weightloading](05_weightloading) |
| 6    | Finetuning LLMs              | This section will introduce LLM finetuning techniques, and we will prepare a small dataset for instruction finetuning, which we will then use to finetune an LLM in LitGPT. | [06_finetuning](06_finetuning)       |

(The code material is based on my [Build a Large Language Model From Scratch](http://mng.bz/orYv) book and also uses the [LitGPT](https://github.com/Lightning-AI/litgpt) library.)

