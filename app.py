import torch
import gradio as gr
from transformers import AutoTokenizer, pipeline,AutoModelWithLMHead


prob_model_name = 'deepset/roberta-base-squad2'
prob_pipeline = pipeline('question-answering', model=prob_model_name, tokenizer=prob_model_name)


answerer_model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
tokenizer = AutoTokenizer.from_pretrained(answerer_model_name)
model = AutoModelWithLMHead.from_pretrained(answerer_model_name)


def retrieve_probability(ctx_input, q_input):

    QA_input = {
        'question': q_input,
        'context': ctx_input
        }
    response = prob_pipeline(QA_input)


    return response['score']

def generate_answer(ctx_input, q_input):
    question = q_input
    context = ctx_input
    input = f"question: {question} context: {context}"
    encoded_input = tokenizer([input],
                             return_tensors='pt',
                             max_length=512,
                             truncation=True)
    output = model.generate(input_ids = encoded_input.input_ids,
                            attention_mask = encoded_input.attention_mask,
                            max_length=126,
                            num_beams=5, 
         )

                            # no_repeat_ngram_size=2) # this return only the tokens
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output

def confidence_star(probability_score):
    if probability_score >= 0.8:
        stars= "⭐️" * 5
    elif probability_score >= 0.6:
        stars= "⭐️" * 4
    elif probability_score >= 0.4:
        stars= "⭐️" * 3
    elif probability_score >= 0.2:
        stars= "⭐️" * 2
    else:
        stars= "⭐️"
    return stars


def generate_answer_with_confidence(ctx_input, q_input):
    answer = generate_answer(ctx_input, q_input)
    confidence= retrieve_probability(ctx_input, q_input)
    stars = confidence_star(confidence)

    complete_answer = f"""confidence: {stars}\n\n{answer}"""

    return complete_answer




input_1 = gr.TextArea(label = 'Context')
input_2 = gr.Textbox(label = 'Question')



def chat(context, question, history = []):

    message = question
    response = generate_answer_with_confidence(context, question)


    history.append((message, response))
    return history, history

chatbot = gr.Chatbot().style(color_map=("green", "pink"))


examples = [
    [r"""
    William Shakespeare (April 1564 – 23 April 1616) was an English playwright, 
    poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. 
    He is often called England's national poet and the "Bard of Avon"
    """, "When was Shakespeare born?"],

        [r"""
    William Shakespeare (April 1564 – 23 April 1616) was an English playwright, 
    poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. 
    He is often called England's national poet and the "Bard of Avon"
    """, "Who is Shakespeare?"],

    [r"""
    William Shakespeare (April 1564 – 23 April 1616) was an English playwright, 
    poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. 
    He is often called England's national poet and the "Bard of Avon"
    """, "How was Shakespeare referred as?"],

    [r"""
    William Shakespeare (April 1564 – 23 April 1616) was an English playwright, 
    poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. 
    He is often called England's national poet and the "Bard of Avon"
    """, "Did Shakespeare go to the moon?"],

    [r"""
    Shakespeare produced most of his known works between 1589 and 1613.
    His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres. 
    He then wrote mainly tragedies until 1608, among them Hamlet, Romeo and Juliet, Othello, King Lear, and Macbeth, 
    all considered to be among the finest works in the English language. 
    """, "What are the plays by Shakespeare?"], 

    [r"""
    Shakespeare produced most of his known works between 1589 and 1613.
    His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres. 
    He then wrote mainly tragedies until 1608, among them Hamlet, Romeo and Juliet, Othello, King Lear, and Macbeth, 
    all considered to be among the finest works in the English language. 
    """, "Did Shakespeare write the play 'Les Miserables'?"], 


]

demo = gr.Interface(
    fn = chat,
    inputs= [input_1, input_2, 'state'],
    outputs= [chatbot, 'state'],
    examples = examples,
    allow_flagging="never",
)


demo.launch()
