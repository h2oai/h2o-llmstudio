import pandas as pd
from pandas.core.frame import DataFrame
from sqlalchemy import Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    """Dataset table"""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column("id", Integer, autoincrement=True, primary_key=True)
    name: Mapped[str] = mapped_column("name", String, unique=True)
    path: Mapped[str] = mapped_column("path", String)
    config_file: Mapped[str] = mapped_column("config_file", String)
    train_rows: Mapped[int] = mapped_column("train_rows", Integer)
    validation_rows: Mapped[int | None] = mapped_column(
        "validation_rows", Integer, nullable=True
    )


class Experiment(Base):
    """Experiment table"""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column("id", Integer, primary_key=True)
    name: Mapped[str] = mapped_column("name", String)
    mode: Mapped[str] = mapped_column("mode", String)
    dataset: Mapped[str] = mapped_column("dataset", String)
    config_file: Mapped[str] = mapped_column("config_file", String)
    path: Mapped[str] = mapped_column("path", String)
    seed: Mapped[int] = mapped_column("seed", Integer)
    process_id: Mapped[int] = mapped_column("process_id", Integer)
    gpu_list: Mapped[str] = mapped_column("gpu_list", String)


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
