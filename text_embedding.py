import time
import nltk
from sentence_transformers import SentenceTransformer, models
from transformers import LongformerTokenizer, LongformerModel
from transformers import BigBirdTokenizer, BigBirdModel
from transformers import AutoTokenizer, AutoModel
import torch


class TextEmbedding:
    def __init__(self, documents,model_name,method,max_seq):
        match method:
            case 1:#SL
                self.data = self.embedding_with_SenTran(documents, model_name, method,max_seq)
            case 2:#BL
                self.data = self.embedding_with_Splitting(documents, model_name,max_seq)
            case 3:#Default
                start = time.time()
                self.data = self.embedding_with_SenTran(documents, model_name, method,max_seq)
                end = time.time()
                print(model_name+"creating embedding time {}".format(end - start))
            case 4:  # Longformer
                self.data = self.embedding_with_LongFormer(documents, model_name)
            case 5:  # BigBird
                self.data = self.embedding_with_BigBird(documents, model_name)

    def calculateMeanSentenceEmbedding(self, documents, model,seq_length):
        #Calculate the pragraph embedding by calculating the mean of all sentences embedding in that pragraph
        document_encode_list=[]
        model.max_seq_length = seq_length
        for document in documents:
            sentences=[sentence for sentence in nltk.sent_tokenize(document) if len(sentence)>10]
            if(len(sentences)!=0):
                encoded_sentences=model.encode(sentences)
                document_encode=encoded_sentences.mean(axis=0)
                document_encode_list.append(document_encode)
            else:
                print(document)

        return document_encode_list
    def mean_pooling(self,model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def embedding_with_SenTran(self,documents_, modelname_,method_,max_seq_):
        model=SentenceTransformer(modelname_)
        word_embedding_model = models.Transformer(modelname_)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=True,
                                        pooling_mode_cls_token=False,
                                        pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        if(method_==1):
            seq_length=max_seq_
            embedded_sentences= self.calculateMeanSentenceEmbedding(documents_,model,seq_length)
        elif (method_==3):
            model.max_seq_length = max_seq_
            embedded_sentences = model.encode(documents_,show_progress_bar=True)

        return embedded_sentences
    def embedding_with_Splitting(self, documents,modelname_,max_seq_):
        document_encode_list = []
        tokenizer = AutoTokenizer.from_pretrained(modelname_)
        model = AutoModel.from_pretrained(modelname_)
        for document in documents:
            # Tokenize text
            encoded_input = tokenizer.encode_plus(document, add_special_tokens=False, return_tensors='pt')
            input_id_chunks = encoded_input['input_ids'][0].split(max_seq_-2)
            mask_chunks = encoded_input['attention_mask'][0].split(max_seq_-2)
            #chunk_size=512
            chunk_size=max_seq_
            input_id_chunks=list(input_id_chunks)
            mask_chunks=list(mask_chunks)

            for i in range(len(input_id_chunks)):
                input_id_chunks[i]=torch.cat([torch.Tensor([101]),input_id_chunks[i],torch.Tensor([102])])
                mask_chunks[i]=torch.cat([torch.Tensor([1]),mask_chunks[i],torch.Tensor([1])])
                chunk_length=chunk_size-input_id_chunks[i].shape[0]
                if(chunk_length>0):
                    input_id_chunks[i]=torch.cat([input_id_chunks[i],torch.tensor([0]*chunk_length)])
                    mask_chunks[i]=torch.cat([mask_chunks[i],torch.tensor([0]*chunk_length)])

            input_ids=torch.stack(input_id_chunks)
            attention_mask=torch.stack(mask_chunks)

            input_dict={
                'input_ids':input_ids.long(),
                'attention_mask':attention_mask.int()
            }

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**input_dict)

            # Perform pooling. In this case, mean pooling to find the mean of word embeddings
            mean_of_token_embeddings = self.mean_pooling(model_output, attention_mask)
            document_embeddings =torch.mean(mean_of_token_embeddings, axis=0)
            document_embeddings=document_embeddings.numpy()
            document_encode_list.append(document_embeddings)
        return document_encode_list
    def embedding_with_LongFormer(self,documents,modelname_):
        document_encode_list = []
        tokenizer = LongformerTokenizer.from_pretrained(modelname_)
        model = LongformerModel.from_pretrained(modelname_)
        for document in documents:
            input = tokenizer(document, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                output = model(**input)
                embeddings = output.last_hidden_state[:, 0, :]
                cls_embeddings= embeddings.squeeze().numpy()
            document_encode_list.append(cls_embeddings)

        return document_encode_list
    def embedding_with_BigBird(self,documents,modelname_):
        document_encode_list = []
        tokenizer = BigBirdTokenizer.from_pretrained(modelname_)
        model = BigBirdModel.from_pretrained(modelname_)
        for document in documents:
            input = tokenizer(document, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                output = model(**input)
                embeddings = output.last_hidden_state
                mean_embedding = embeddings.mean(dim=1).squeeze().numpy()
            document_encode_list.append(mean_embedding)
        return document_encode_list




    







