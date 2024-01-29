import pandas as pd
from pandas.core.frame import DataFrame
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import Session

Base: DeclarativeMeta = declarative_base()


class Dataset(Base):
    """Dataset table"""

    __tablename__ = "datasets"

    id = Column("id", Integer, autoincrement=True, primary_key=True)
    name = Column("name", String, unique=True)
    path = Column("path", String)
    config_file = Column("config_file", String)
    train_rows = Column("train_rows", Integer)
    validation_rows = Column("validation_rows", Integer)


class Experiment(Base):
    """Experiment table"""

    __tablename__ = "experiments"

    id = Column("id", Integer, primary_key=True)
    name = Column("name", String)
    mode = Column("mode", String)
    dataset = Column("dataset", String)
    config_file = Column("config_file", String)
    path = Column("path", String)
    seed = Column("seed", Integer)
    process_id = Column("process_id", Integer)
    gpu_list = Column("gpu_list", String)


class Database:
    """Class for managing database."""

    def __init__(self, path_db: str) -> None:
        """Initialize database

        Args:
            path_db: path to sqlite database file
        """

        self.__engine__ = create_engine(f"sqlite:///{path_db}")
        Base.metadata.create_all(self.__engine__)
        self._session = Session(self.__engine__)

    def add_dataset(self, dataset: Dataset) -> None:
        """Add a dataset to the table

        Args:
            dataset: dataset to add
        """
        self._session.add(dataset)
        self._session.commit()

    def delete_dataset(self, id: int) -> None:
        """Delete a dataset from the table

        Args:
            id: dataset id to delete
        """

        dataset = self._session.query(Dataset).get(int(id))
        self._session.delete(dataset)
        self._session.commit()

    def get_dataset(self, id: int) -> Dataset:
        """Return dataset given an id

        Args:
            id: dataset id to return

        Returns:
            Dataset with given id
        """

        return self._session.query(Dataset).get(int(id))

    def get_datasets_df(self) -> DataFrame:
        """Return dataframe containing all datasets

        Returns:
            All datasets
        """

        datasets = pd.read_sql(self._session.query(Dataset).statement, self.__engine__)
        return datasets.sort_values("id", ascending=False)

    def add_experiment(self, experiment: Experiment) -> None:
        """Add an experiment to the table

        Args:
            experiment: experiment to add
        """

        self._session.add(experiment)
        self._session.commit()

    def delete_experiment(self, id: int) -> None:
        """Delete an experiment from the table

        Args:
            id: experiment id to delete
        """

        experiment = self._session.query(Experiment).get(int(id))
        self._session.delete(experiment)
        self._session.commit()

    def get_experiment(self, id: int) -> Experiment:
        """Return experiment given an id

        Args:
            id: experiment id to return

        Returns:
            Experiment with given id
        """

        return self._session.query(Experiment).get(int(id))

    def get_experiments_df(self) -> DataFrame:
        """Return dataframe containing all experiments

        Returns:
            All experiments
        """

        experiments = pd.read_sql(
            self._session.query(Experiment).statement, self.__engine__
        )
        return experiments.sort_values("id", ascending=False)

    def rename_experiment(self, id: int, new_name: str, new_path: str) -> None:
        """Rename experiment given id and new name

        Args:
            id: experiment id
            new_name: new name
        """

        experiment = self.get_experiment(id)
        experiment.name = new_name
        experiment.path = new_path
        self._session.commit()

    def update(self) -> None:
        self._session.commit()
