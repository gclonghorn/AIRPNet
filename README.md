# PIRNet-plus

## Environment

This project was developed and tested with the following environment:

- CUDA: 11.8
- Python: 3.8.13
- PyTorch: 2.0.1

## Checkpoints

You can access the checkpoints for PIRNet++ at the following URL:
- Checkpoint: [Link to Checkpoint](https://drive.google.com/drive/folders/1mbcB_SVK3beaxeRQYXtkHuwEQ4IT_OMX?usp=sharing)

## Datasets

You can access the datasets for PIRNet++ at the following URL:
- Datasets: [Link to Datasets](https://drive.google.com/drive/folders/1xmY6J-QJsseygOkwcd3lqDCMDI4jFMp0?usp=drive_link)


## Running Instructions

To run the tests, execute the following command:

```bash
bash test.sh
```

To train the LIH model, use the following command:

```bash
bash train_LIH.sh
```

To train the LSR model, use the following command:

```bash
bash train_LSR.sh
```

Please ensure to update the paths for the datasets and checkpoints to match your local setup in the respective scripts.
