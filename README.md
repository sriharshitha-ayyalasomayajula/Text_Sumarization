Text Summarization with Seq2Seq and Transformer Models 

DataSource: CNN daily news dataset - [CNN data] (https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail)  -Click on this link and download the tokenized data
Turning Red movie plot - we have taken this data form their wikipedia page - [Turning Red] (https://en.wikipedia.org/wiki/Turning_Red)

Language: Python  (ver 3.11). 
Platform:  Google Colab - (colab.research.com) Login using PDX gmail credentials.    
Software Requirements:  MacOS or Windows 10.  
Steps:
Experiment 1: Seq2Seq model with CNN/ daily mail data - 
1. Open a new notebook in your colab and connect with a GPU. 
2. Upload all the data into your drive.
3. Now, connect your google drive and the working notebook (mount). 
4. Import the necessary libraries.
5. Load the data by passing the path of the folder in which the data is stored in your drive and split it to test and train. 
6. Now that the data is loaded, We do pre-processing by dropping the id column, defining the contradictions, cleaning and trimming, and removing the stop words which is removing the unnecessary noise from the data which might impact the scores. 
7. Then we tokenize the cleaned data and validate it. 
8. We then define the embedding layer followed by encoder and decoder with the LSTM component. 
9. We then compile the model, define a function to generate a summary using the define model. 
10.Then we test it using the testing data and save the results to a csv file in the same folder in which the data is stored. 
11. We  apply rogue scores to the model.

Result: We found that the summaries weren’t accurate and meaningful and the rogue score was less compared to the state-of-the-art seq2seq model.

Experiment 2: Text Summarization with Transformers
1. Open a new notebook in colab and connect with GPU 
2. Install and import all the required libraries - Sumy, nltk, transformers, pipelines etc.
3. Here, the input is a plain text of the Turning Red movie’s plot (you can give any other input text).
4. For summarization using sumy library - we consider LSA and LexRank - to implement that, we have to first parse the text and then apply the summarizers. 
5. Then for summarizations using transformers - after importing and installing transformers and pipelines that are provided by “hugging face” - we just need to call any summarization model using the pipeline. 
6. First we applied the default model i.e., distill bart model 
Syntax:  summarizer = pipeline("summarization")
Then pass the input text to this summarizer to generate a summary. 
7. Second, we applied pegasus-xsum model
Syntax: pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum", tokenizer="google/pegasus-xsum", framework="pt")
Then pass the input text to this summarizer to generate a summary.
8. Third, we applied bart-large model
Syntax:  bart_summarizer = pipeline("summarization", model="facebook/bart-large", tokenizer="facebook/bart-large", framework="tf")
Then pass the input text to this summarizer to generate a summary.
9. Fourth, we applied mbart model
Syntax: mbart_summarizer = pipeline ("summarization", model ="facebook/mbart-large-50", tokenizer = "facebook/mbart-large-50",use_fast = False)
Then pass the input text to this summarizer to generate a summary.
10. We also apply rogue scores to all these models. 
11. We then tabulate the results -[ model, summary]

Result:  We found that the pegasus-xsum model had a better rogue score than the others and its summary was brief and meaningful followed by the default model’s summary. 

To Do:

* For experiment 1, even though we have applied a seq2seq LSTM based model, the accuracy that has been generated was not meaningful and the rogue scores were less compared to the current models. To address this we would like to apply an attention layer to our model, train it and generate summaries. 
* For experiment 2 , we have implemented transformers using the “hugging face” library and generated summaries. Through this experiment, we found that training a transformer kind model over a simple text and large data requires a machine which has a better computing power, we need to use powerful GPUs. We have implemented these using Colab which gives us access to GPU’s even then it was quite difficult to run the models and train them over data. So, in the future we would like to use NVIDIAs  cuda GPUs for this task and also implement a model and its inference instead of just using inferences. 


