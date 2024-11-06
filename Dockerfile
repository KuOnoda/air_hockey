# ベースイメージとして Miniconda を使用
FROM continuumio/miniconda3

# 作業ディレクトリの設定
WORKDIR /app

# environment.yml をコンテナにコピー
COPY environment.yml .

# 環境の作成とアクティベート
RUN conda env create -f environment.yml
RUN echo "conda activate challenge" >> ~/.bashrc

# デフォルトのシェルを指定の conda 環境で起動
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "<your_env_name>", "bash"]
