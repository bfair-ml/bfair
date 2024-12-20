FROM jpconsuegra/autogoal

RUN curl -fsSL https://starship.rs/install.sh > ~/starship.sh
RUN sh ~/starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc
RUN rm ~/starship.sh

RUN pip install -U black

RUN pip install -U streamlit==0.83

RUN pip install scikit-learn

RUN pip install seaborn

RUN python -c "import nltk; nltk.download('stopwords')"

RUN python -m spacy download xx_ent_wiki_sm

RUN cd /usr/lib/python3/dist-packages/ \
 && sudo git clone https://github.com/huggingface/neuralcoref.git \
 && cd neuralcoref \
 && sudo pip install -r requirements.txt \
 && sudo pip install -e .

RUN python -m spacy download en_core_web_sm-2.3.1 --direct
RUN python -m spacy download es_core_news_sm-2.3.1 --direct # Lemmatization in newer versions of spacy (ES) do not turn femenine words into masculine

RUN pip install datasets
RUN mkdir -p /home/coder/huggingface_data/
ENV HF_HOME="/home/coder/huggingface_data/"

WORKDIR /home/coder/bfair

CMD [ "make", "dashboard"]