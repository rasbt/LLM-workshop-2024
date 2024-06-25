# LLM-workshop-2024



## Overview

This tutorial is aimed at coders interested in understanding the building blocks of large language models (LLMs), how LLMs work, and how to code them from the ground up in PyTorch. We will kick off this tutorial with an introduction to LLMs, recent milestones, and their use cases. Then, we will code a small GPT-like LLM, including its data input pipeline, core architecture components, and pretraining code ourselves. After understanding how everything fits together and how to pretrain an LLM, we will learn how to load pretrained weights and finetune LLMs using open-source libraries.

## Setup instructions

A ready-to-go cloud environment, complete with all code examples and dependencies installed, will be shared during the workshop. This will enable participants to run all code, particularly in the pretraining and finetuning sections, on a GPU.

In addition, see the instructions in the [setup](./setup) folder to set up your computer to run the code locally.

## Outline

|      | Title                        | Description                                                  | Code |
| ---- | ---------------------------- | ------------------------------------------------------------ | ---- |
| 1    | Introduction to LLMs         | An introduction to the workshop introducing LLMs, the topics being covered in this workshop, and setup instructions. | TBD  |
| 2    | Understanding LLM Input Data | In this section, we are coding the text input pipeline by implementing a text tokenizer and a custom PyTorch DataLoader for our LLM | TBD  |
| 3    | Coding an LLM architecture   | In this section, we will go over the individual building blocks of LLMs and assemble them in code. We will not cover all modules in meticulous detail but will focus on the bigger picture and how to assemble them into a GPT-like model. | TBD  |
| 4    | Pretraining LLMs             | In part 4, we will cover the pretraining process of LLMs and implement the code to pretrain the model architecture we implemented previously. Since pretraining is expensive, we will only pretrain it on a small text sample available in the public domain so that the LLM is capable of generating some basic sentences. | TBD  |
| 5    | Loading pretrained weights   | Since pretraining is a long and expensive process, we will now load pretrained weights into our self-implemented architecture. Then, we will introduce the LitGPT open-source library, which provides more sophisticated (but still readable) code for training and finetuning LLMs. We will learn how to load weights of pretrained LLMs (Llama, Phi, Gemma, Mistral) in LitGPT. | TBD  |
| 6    | Finetuning LLMs              | This section will introduce LLM finetuning techniques, and we will prepare a small dataset for instruction finetuning, which we will then use to finetune an LLM in Lit-GPT. | TBD  |

(The code material is loosely based on my [Build a Large Language Model From Scratch](http://mng.bz/orYv) book and also uses the [LitGPT](https://github.com/Lightning-AI/litgpt) library)
