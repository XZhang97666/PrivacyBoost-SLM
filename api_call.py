import os
import openai
import argparse
import json
from tqdm.auto import tqdm
from datasets import load_dataset
openai.api_key = os.getenv("OPENAI_API_KEY")
import random



def parse_args():
    parser = argparse.ArgumentParser(description="Generate Reasoning Path from OpenAI-GPT3")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_idx", type=int, default=0, help="Generate reasoning path for top k examples")
    parser.add_argument("--end_idx", type=int, default=1, help="Generate reasoning path for top k examples")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for GPT-3 generation")
    parser.add_argument("--output_dir", type=str, default="./data/GPT3")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--stop_token", type=str, default="\n")
    parser.add_argument("--batch_prompt_size", type=int, default=1, help="batch size for batch_prompting ")
    args = parser.parse_args()
    return args
async def main():
    args = parse_args()

    intro_prompt=""
    if(args.dataset_name=="csqa"):
        if "context" in args.mode:
            intro_prompt="Generate a overall context based on commonsense knowledge by utilizing the provided keywords and answer choices. Then, generate the specific commonsense knowledge context for each candidate answers and identify the relationship between it and the overall context. If the relationship is not exist, generate 'No relationship can be found.' as the last sentence.\n\n"
            with open("./data/csqa/apicall/"+args.split+"_concepts.json") as f:
                concepts = json.load(f)
        dataset = load_dataset('commonsense_qa',split=args.split)
        with open("./data/csqa/apicall/prompts.json") as f:
            init_prompt_dict = json.load(f)
        if 'context' in args.mode:
            init_prompt=init_prompt_dict['Context']
    elif(args.dataset_name=="obqa"):
        if "context" in args.mode:
            intro_prompt="Generate a overall context based on commonsense knowledge by utilizing the provided keywords and answer choices. Then, generate the specific commonsense knowledge context for each candidate answers and identify the relationship between it and the overall context. If the relationship is not exist, generate 'No relationship can be found.' as the last sentence.\n\n"
            with open("./data/obqa/apicall/"+args.split+"_concepts.json") as f:
                concepts = json.load(f)
        with open("./data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/"+args.split+"_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        with open("./data/obqa/apicall/prompts.json") as f:
            init_prompt_dict = json.load(f)
        if 'context' in args.mode:
            init_prompt=init_prompt_dict['Context']
    elif(args.dataset_name=="medqa"):
        with open("./data/MedQA/questions/US/4_options/phrases_no_exclude_"+args.split+".jsonl")  as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        if "context" in args.mode:
            if 'span' in args.mode:
                intro_prompt="Generate a overall context using medical knowledge by incorporating the given random span extracted from the question and candidate answers. Then, generate the specific medical knowledge-intensive context of each candidate answers and identify relationship between it and the overall context. If the relationship is not exist, generate 'No relationship can be found.' as the last sentence.\n\n"
            else:
                intro_prompt="Generate a overall context using medical knowledge by incorporating the given keywords extracted from the question and candidate answers. Then, generate the specific medical knowledge-intensive context of each candidate answers and identify relationship between it and the overall context. If the relationship is not exist, generate 'No relationship can be found.' as the last sentence.\n\n"

            if '25' in args.mode:
                with open("./data/MedQA/apicall/"+args.split+"_concepts25.json") as f:
                    concepts = json.load(f)
            elif '50' in args.mode:
                with open("./data/MedQA/apicall/"+args.split+"_concepts50.json") as f:
                    concepts = json.load(f)
            elif '75' in args.mode:
                with open("./data/MedQA/apicall/"+args.split+"_concepts75.json") as f:
                    concepts = json.load(f)
            elif 'randomspan' in args.mode:
                with open("./data/MedQA/apicall/"+args.split+"_span_random.json") as f:
                    concepts = json.load(f)
            elif 'random' in args.mode:
                with open("./data/MedQA/apicall/"+args.split+"_concepts_random.json") as f:
                    concepts = json.load(f)
            else:
                with open("./data/MedQA/apicall/"+args.split+"_concepts.json") as f:
                    concepts = json.load(f)

        if '25' in args.mode:
            with open("./data/MedQA/apicall/prompts25.json") as f:
                init_prompt_dict = json.load(f)
        elif '50' in args.mode:
            with open("./data/MedQA/apicall/prompts50.json") as f:
                init_prompt_dict = json.load(f)
        elif '75' in args.mode:
            with open("./data/MedQA/apicall/prompts75.json") as f:
                init_prompt_dict = json.load(f)
        elif 'randomspan' in args.mode:
            with open("./data/MedQA/apicall/prompts_randomspan.json") as f:                                                                                                                                         
                init_prompt_dict = json.load(f)
        elif 'random' in args.mode:
            with open("./data/MedQA/apicall/prompts_random.json") as f:                                                                                                                                         
                init_prompt_dict = json.load(f)
        else:
            with open("./data/MedQA/apicall/prompts.json") as f:
                init_prompt_dict = json.load(f)
            
        init_prompt = init_prompt_dict['Context']
        



    elif(args.dataset_name=="mmlu"):

        dataset = load_dataset('./data/mmlu/professional_medicine/',split=args.split)

        if "context" in args.mode:
            intro_prompt=intro_prompt="Generate a overall context using medical knowledge by incorporating the given keywords extracted from the question and candidate answers. Then, generate the specific medical knowledge-intensive context of each candidate answers and identify relationship between it and the overall context. If the relationship is not exist, generate 'No relationship can be found.' as the last sentence.\n\n"

            with open("./data/mmlu/apicall/"+args.split+"_concepts.json") as f:
                concepts = json.load(f)
          
            with open("./data/MedQA/apicall/prompts.json") as f:
                init_prompt_dict = json.load(f)
            init_prompt = init_prompt_dict['Context']

    elif(args.dataset_name=="medmcqa"):

        dataset = load_dataset("medmcqa",split=args.split)

        if args.split=="train":
            with open("./data/medmc/indexes_list_file.json","r") as f:
                idx_list=json.load(f)

        if "context" in args.mode:
            intro_prompt="Generate a overall context using medical knowledge by incorporating the given keywords extracted from the question and candidate answers. Then, generate the specific medical knowledge-intensive context of each candidate answers and identify relationship between it and the overall context. If the relationship is not exist, generate ' No relationship can be found.' as the last sentence.\n\n"

            with open("./data/medmc/apicall/"+args.split+"_concepts.json") as f:
                concepts = json.load(f)

            with open("./data/medmc/apicall/prompts.json") as f:
                init_prompt_dict = json.load(f)
            init_prompt = init_prompt_dict['Context']
    
    elif(args.dataset_name=="headqa"):

        dataset = load_dataset("head_qa","en",split=args.split)

        if "context" in args.mode:
            intro_prompt="Generate a overall context using medical knowledge by incorporating the given keywords extracted from the question and candidate answers. Then, generate the specific medical knowledge-intensive context of each candidate answers and identify relationship between it and the overall context. If the relationship is not exist, generate 'No relationship can be found.' as the last sentence.\n\n"

            with open("./data/headqa/apicall/"+args.split+"_concepts.json") as f:
                concepts = json.load(f)
            # with each answer
           
            with open("./data/headqa/apicall/prompts.json") as f:
                init_prompt_dict = json.load(f)
            init_prompt = init_prompt_dict['Context']
        elif 'cot' in args.mode:
            intro_prompt="'The following are multiple choice questions (with answers) about medical knowledge.\n\n"
            # with each answer
            with open("./data/headqa/apicall/cots.json") as f:
                init_prompt_dict = json.load(f)
            init_prompt = init_prompt_dict['Explanation']


    data = []
    total = 0
    batch_prompt_idx=0
    args.end_idx = min(args.end_idx,len(dataset))
    context_key='Context'
    import pdb;pdb.set_trace()
    progress_bar = tqdm(range(args.start_idx,args.end_idx), disable=False)
    for i in range(args.start_idx, args.end_idx,args.batch_prompt_size):

        if(args.dataset_name=="csqa"):

            keywords="Question Keywords: "
            answer_choices = "Candidate Answers: "
            keywords+=", ".join(concepts[i]['Question Keywords'])
            for j in range(len(dataset[i]['choices']["label"])):
                answer_choices += "("+dataset[i]['choices']["label"][j].lower()+") "+dataset[i]['choices']["text"][j]+" "
            cur_prompt = keywords+"\n"+ answer_choices+"\n"+ "Context:"
            prompt = intro_prompt+init_prompt + cur_prompt
            if 'chatgpt' in args.mode:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system","content":"As a knowledgeable assistant, your role is to generate commonsense information in accordance with the provided instructions and examples."},
                        {"role": "user", "content": prompt}
                        ],
                        temperature=args.temperature,
                        max_tokens=500,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[args.stop_token]
                    )
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = response['choices'][0]['message']
                    data_i["answer"] = "(" + dataset[i]['answerKey'].lower()+ ")"


                except:
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = "GPT-3 Api Call Error"
                    data_i["answer"] = "(" + dataset[i]['answerKey'].lower()+ ")"


        elif(args.dataset_name=="obqa"):
            keywords="Question Keywords: "
            answer_choices = "Candidate Answers: "
            keywords+=", ".join(concepts[i])
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+" "
            cur_prompt = keywords+"\n"+ answer_choices+"\n"+ "Context:"
            prompt = intro_prompt+init_prompt + cur_prompt
            data_i = {}
            data_i["id"] = i
            data_i["question"] = cur_prompt
            data_i["prompt"] = prompt
            data_i["answer"] = "(" + dataset[i]['answerKey'].lower()+ ")"
            if 'chatgpt' in args.mode:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system","content":"As a knowledgeable assistant, your role is to generate commonsense information in accordance with the provided instructions and examples."},
                        {"role": "user", "content": prompt}
                        ],
                        temperature=args.temperature,
                        max_tokens=500,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[args.stop_token]
                    )
                    data_i["output"] = response['choices'][0]['message']


                except:
                    data_i["output"] = "GPT-3 Api Call Error"

        elif (args.dataset_name=="medqa"):
            gname="Context"
            if 'span' in args.mode:
                span="Question Random Span: "
                answer_choices = "Candidate Answers: "
                span+=concepts[i]['span']
                for choice,text in dataset[i]['options'].items():
                    answer_choices += "("+choice.lower()+") "+ text +" "
                cur_prompt = span+"\n"+ answer_choices+"\n"+ gname+":"
            
            else:
                keywords="Question Keywords: "
                answer_choices = "Candidate Answers: "
                keywords+=", ".join(concepts[i]['keywords'])
                for choice,text in dataset[i]['options'].items():
                    answer_choices += "("+choice.lower()+") "+ text +" "
                cur_prompt = keywords+"\n"+ answer_choices+"\n"+ gname+":"
            prompt = intro_prompt+init_prompt + cur_prompt
            if 'chatgpt' in args.mode:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system","content":"As a knowledgeable assistant, your role is to generate medical information in accordance with the provided instructions and examples."},
                         {"role": "user", "content": prompt}
                        ],
                        temperature=args.temperature,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[args.stop_token]
                    )
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = response['choices'][0]['message']
                    data_i["answer"] = "(" + dataset[i]['answer_idx'].lower()+ ")"
                    data_i['meta_info']=dataset[i]['meta_info']
                    data_i['metamap_phrases']=dataset[i]['metamap_phrases']

                except:
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = "GPT-3 Api Call Error"
                    data_i["answer"] = "(" + dataset[i]['answer_idx'].lower()+ ")"
                    data_i['meta_info']=dataset[i]['meta_info']
                    data_i['metamap_phrases']=dataset[i]['metamap_phrases']

          
        elif (args.dataset_name=="medmcqa"):
            ending_names = [elm for elm in dataset.features.keys() if elm.startswith('op')]
            num_choices = len(ending_names)
            endings_dict = {ending_names[e]: chr(ord('a') + e) for e in range(num_choices)}
            if args.split=="train" and "error" not in args.prefix:
                j=idx_list[i]
            else:
                j=i

            gname="Context"
            keywords="Question Keywords: "
            answer_choices = "Candidate Answers: "
            keywords+=", ".join(concepts[j]['keywords'])
            for ending in ending_names:
                answer_choices += "("+endings_dict[ending]+") "+dataset[j][ending] +" "
            cur_prompt = keywords+"\n"+ answer_choices+"\n"+ gname+":"

            prompt = intro_prompt+init_prompt + cur_prompt
            if 'chatgpt' in args.mode:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system","content":"As a knowledgeable assistant, your role is to generate medical information in accordance with the provided instructions and examples."},
                         {"role": "user", "content": prompt}
                        ],
                        temperature=args.temperature,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[args.stop_token]
                    )
                    data_i = {}
                    data_i["id"] = j
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = response['choices'][0]['message']
                    data_i["answer"] = "(" + endings_dict[ending_names[dataset[j]['cop']]]+ ")"
                    data_i['subject_name']=dataset[j]['subject_name']
                    data_i['topic_name']=dataset[j]['topic_name']

                except:
                    data_i = {}
                    data_i["id"] = j
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = "GPT-3 Api Call Error"
                    data_i["answer"] = "(" + endings_dict[ending_names[dataset[j]['cop']]]+ ")"
                    data_i['subject_name']=dataset[j]['subject_name']
                    data_i['topic_name']=dataset[j]['topic_name']

         

        elif (args.dataset_name=="mmlu"):
            num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
            ending_names = [f"ending{i}" for i in range(num_choices )]
            endings_dict = {ending_names[i]: chr(ord('a') + i) for i in range(num_choices)}

            gname="Context"
           
            keywords="Question Keywords: "
            answer_choices = "Candidate Answers: "
            keywords+=", ".join(concepts[i]['keywords'])
            for ending in ending_names:
                answer_choices += "("+endings_dict[ending]+") "+ dataset[i][ending] +" "
            cur_prompt = keywords+"\n"+ answer_choices+"\n"+ gname+":"

            prompt = intro_prompt+init_prompt + cur_prompt
            if 'chatgpt' in args.mode:

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system","content":"As a knowledgeable assistant, your role is to generate medical information in accordance with the provided instructions and examples."},
                         {"role": "user", "content": prompt}
                        ],
                        temperature=args.temperature,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[args.stop_token]
                    )
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = response['choices'][0]['message']
                    data_i["answer"] = "(" + endings_dict[ending_names[dataset[i]['label']]]+ ")"

                except:
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = "GPT-3 Api Call Error"
                    data_i["answer"] = "(" + endings_dict[ending_names[dataset[i]['label']]]+ ")"

        
        elif (args.dataset_name=="headqa"):
            answers=dataset[i]['answers']
            num_choices = len(answers)

            if 'context' in args.mode:
                gname="Context"

                keywords="Question Keywords: "
                answer_choices = "Candidate Answers: "
                keywords+=", ".join(concepts[i]['keywords'])
                for ans in answers:
                    j=ans['aid']
                    text=ans['atext']
                    choice=chr(ord('a')+j-1)
                    answer_choices += "("+choice+") "+text +" "
                cur_prompt = keywords+"\n"+ answer_choices+"\n"+ gname+":"
            else:
                gname='Explanation'

                question="Question: "
                question+=dataset[i]['qtext']
                answer_choices = "Candidate Answers: "
                for ans in answers:
                    j=ans['aid']
                    text=ans['atext']
                    choice=chr(ord('a')+j-1)
                    answer_choices += "("+choice+") "+text +" "

                cur_prompt = question+"\n"+ answer_choices+"\n"+ gname+":"

            prompt = intro_prompt+init_prompt + cur_prompt
            if 'chatgpt' in args.mode:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system","content":"As a knowledgeable assistant, your role is to generate medical information in accordance with the provided instructions and examples."},
                         {"role": "user", "content": prompt}
                        ],
                        temperature=args.temperature,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[args.stop_token]
                    )
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = response['choices'][0]['message']
                    data_i["answer"] = "(" + chr(ord('a')+dataset[i]['ra']-1)+ ")"
                    data_i['category']=dataset[j]['category']

                except:
                    data_i = {}
                    data_i["id"] = i
                    data_i["question"] = cur_prompt
                    data_i["prompt"] = prompt
                    data_i["output"] = "GPT-3 Api Call Error"
                    data_i["answer"] = "(" + chr(ord('a')+dataset[i]['ra']-1)+ ")"
                    data_i['category']=dataset[j]['category']

         

        total += 1
        data.append(data_i)
        progress_bar.update(1)
    meta_string = args.output_dir +"/"+ args.dataset_name
    os.makedirs(meta_string, exist_ok=True)
    with open(os.path.join(meta_string, args.prefix +"-"+ args.mode + "-" + args.split+"-"+str(args.start_idx)+"-"+str(args.end_idx)+"-"+str(args.temperature)+".json"), "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
