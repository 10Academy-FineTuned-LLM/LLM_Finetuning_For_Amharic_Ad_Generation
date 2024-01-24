# LLM Finetuning For Amharic Ad Generation

## About <a name = "about"></a>

This project aims to fine-tune an llm so that it can understand the Amharic language and create an Advertisement in Amharic given a brand information, product brief. It'll utilize messages exported from 25 publicly available channels to extend the pre-training phase of the model as well as fine-tune the model to generate ads later on.

## Usage <a name = "usage"></a>

At this point in time, you'll need the raw data of the channel messages in a directory named data/raw. Then you can follow the following steps to clean the data and make it appropriate for the model:

- `pip install -r requirements.txt`
- inside `parse_and_save.ipynb`, run the function `process_raw_data` to get only the necessary data from the raw data which are id, text, date
- inside `cleaning.ipynb` run the function `clean_parsed_data` to get the cleaned data which has removed emojis, symbols, newlines, extra spaces

## References

- [Medium Article showcasing how Garri Logistics team fine-tuned Llama to understand Amharic](https://medium.com/@garrilogistics/llama-2-amharic-llms-for-low-resource-languages-d6fb0ba332f4)
- [Key terms and concepts](https://osanseviero.github.io/hackerllama/blog/posts/hitchhiker_guide/)
- [Fine-tune lama step by step](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
