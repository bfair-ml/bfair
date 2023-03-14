FROM jpconsuegra/autogoal

RUN curl -fsSL https://starship.rs/install.sh > ~/starship.sh
RUN sh ~/starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc
RUN rm ~/starship.sh

RUN pip install -U black

RUN pip install -U streamlit==0.83

RUN pip install sklearn

RUN pip install seaborn

WORKDIR /home/coder/bfair

CMD [ "make", "dashboard"]