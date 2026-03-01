# Seq2Seq English–French Dataset

## Overview
This repository contains the English–French parallel dataset used for a Sequence-to-Sequence (Seq2Seq) model.

## Dataset Description
- File: `eng-fra.txt`
- Format: Tab-separated sentence pairs
- Each line contains:
	- English sentence
	- French sentence

Example:
Go.    Va !
Run!   Cours !

## Train/Test Split
The dataset is split into:
- 80% Training Data
- 20% Testing Data

Split performed using `train_test_split` from sklearn with `random_state=42`.

## Author
Sravani B
