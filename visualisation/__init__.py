import pandas as pd
import typing as tp
import ipywidgets as widgets
import numpy as np
import pandas as pd

from IPython.display import display
from ipywidgets.embed import embed_minimal_html
from ipywidgets.widgets import widget_selectioncontainer
from pydantic import BaseModel



def image_html(item_id: int) -> str:
    return f"<img src='https://media.tv4.live/{item_id}.movie.poster.jpg' style=max-height:150px;/>"


def bold_html_rounded(score: int) -> str:
    return f"<p style='color:#3B9C9C;'>{round(score, 2)}</p>"


class ShowcaseOptions(BaseModel):
    item_df_columns: tp.List[str]
    item_df_renaming: tp.Dict[str, str]
    formatters: tp.Dict[str, tp.Callable]


PROJECT_OPTIONS = ShowcaseOptions(
    item_df_columns=[
        "item_id",
        "title",
        "genres",
        #"countries",
        "release_year",
        "watched_in_all_time",
    ],
    item_df_renaming={"watched_in_all_time": "watches", "item_id": "img"},
    formatters=dict(img=image_html, score=bold_html_rounded),
)


class Columns:
    """
    Fixed column names for tables that contain interactions and recommendations.
    """

    User = "user_id"
    Item = "item_id"
    Date = "date"
    Model = "model"
    Rank = "rank"


COLUMNS = Columns()


class DataNames:
    """
    Fixed names for saving data files
    """

    Interactions = "interactions.csv"
    Recos = "full_recos.csv"
    UsersDict = "users_dict.csv"
    Truth = "ground_truth.csv"
    Items = "items_data.csv"


DATANAMES = DataNames()


class ItemTypes:
    """
    Fixed names for item types
    """

    Viewed = "viewed"
    Recos = "recos"
    Truth = "ground_truth"


ITEMTYPES = ItemTypes()


class ShowcaseDataStorage:
    """
    Helper class to hold all data for showcase purposes.
    - Holds info about interactions, recommendations and (if provided) ground truth  for users.
    - Downloads additional info about items.
    - Holds `users_dict` top map user ids with user names.
    - Supports adding random users to `users_dict`.
    - Supports removing exceeding data that is not needed to display users from `user_dict` and their items.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        full_recos: pd.DataFrame,
        users_dict: tp.Dict[str, tp.Any],
        item_data: tp.Optional[pd.DataFrame],
        ground_truth: tp.Optional[pd.DataFrame] = None,
        n_add_random_users: int = 0,
        remove_exceeding_data: tp.Optional[bool] = True,
        convert_ids_to_int: bool = True,
    ) -> None:
        self.users_dict = users_dict
        self.interactions = interactions
        if COLUMNS.Model not in full_recos.columns:
            if "model_name" in full_recos.columns:
                full_recos.rename(columns={"model_name": COLUMNS.Model}, inplace=True)
            else:
                full_recos[COLUMNS.Model] = "Reco_Model"
        self.full_recos = full_recos
        self.ground_truth = ground_truth
        if COLUMNS.Model not in full_recos.columns:
            print(full_recos.columns)
        self.model_names = full_recos[COLUMNS.Model].unique()
        self.item_data = item_data
        self.exceeding_data_removed = False
        if n_add_random_users > 0:
            self.update_users_with_random(n=n_add_random_users)
        if remove_exceeding_data:
            self.remove_exceeding_data()
        if convert_ids_to_int:
            for df in [
                self.interactions,
                self.full_recos,
                self.ground_truth,
                self.item_data,
            ]:
                if isinstance(df, pd.DataFrame):
                    self._convert_df_id_cols_to_int(df)

    def _convert_df_id_cols_to_int(self, df):
        if Columns.User in df.columns:
            df[Columns.User] = df[Columns.User].astype("int64")
        if Columns.Item in df.columns:
            df[Columns.Item] = df[Columns.Item].astype("int64")
        if Columns.Rank in df.columns:
            df[Columns.Rank] = df[Columns.Rank].astype("int32")

    def get_relevant_items(self) -> np.ndarray:
        inter_items = self.interactions[COLUMNS.Item].unique()
        recos_items = self.full_recos[COLUMNS.Item].unique()
        all_items = np.union1d(inter_items, recos_items)
        if self.ground_truth is not None:
            truth_items = self.ground_truth[COLUMNS.Item].unique()
            all_items = np.union1d(all_items, truth_items)
        return all_items

    def get_user_names(self) -> tp.List[str]:
        return [*self.users_dict.keys()]

    def get_user_idx(self) -> tp.List[str]:
        return [*self.users_dict.values()]

    def get_viewed_items_for_user(self, user_id: tp.Any) -> np.ndarray:
        user_interactions = self.interactions[
            self.interactions[COLUMNS.User] == user_id
        ]
        return user_interactions[COLUMNS.Item].unique()

    def get_recos_for_user(self, user_id: tp.Any, model_name: str) -> np.ndarray:
        if model_name not in self.model_names:
            raise ValueError(f"{model_name} not in model names: {self.model_names}")
        model_recos = self.full_recos[
            (self.full_recos[COLUMNS.Model] == model_name)
            & (self.full_recos[COLUMNS.User] == user_id)
        ]
        return model_recos[COLUMNS.Item].unique()

    def get_ground_truth_for_user(self, user_id: tp.Any) -> np.ndarray:
        if self.ground_truth is None:
            raise TypeError("Ground truth not specified")
        user_truth = self.ground_truth[self.ground_truth[COLUMNS.User] == user_id]
        return user_truth[COLUMNS.Item].unique()

    def update_users_with_random(self, n: int = 10) -> None:
        if self.exceeding_data_removed:
            raise TypeError(
                "Not possible to select more users since exceeding data was removed"
            )
        if self.ground_truth is None:
            all_users = self.full_recos[COLUMNS.User].unique()
        else:
            truth_users = self.ground_truth[COLUMNS.User].unique()
            recos_users = self.full_recos[COLUMNS.User].unique()
            all_users = np.intersect1d(truth_users, recos_users, assume_unique=True)
        new_idx = np.random.choice(all_users, size=n, replace=False)
        new_users_dict = {f"random_{i}": new_idx[i] for i in range(n)}
        self.users_dict.update(new_users_dict)

    def remove_exceeding_data(self) -> None:
        relevant_users = self.get_user_idx()
        self.interactions = self.interactions[
            self.interactions[COLUMNS.User].isin(relevant_users)
        ].copy()
        self.full_recos = self.full_recos[
            self.full_recos[COLUMNS.User].isin(relevant_users)
        ].copy()
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth[
                self.ground_truth[COLUMNS.User].isin(relevant_users)
            ].copy()
        relevant_items = self.get_relevant_items()
        if isinstance(self.item_data, pd.DataFrame):
            self.item_data = self.item_data[
                self.item_data[COLUMNS.Item].isin(relevant_items)
            ].copy()
        else:
            raise TypeError("Item data was not specified")
        self.exceeding_data_removed = True


class Showcase(ShowcaseDataStorage):
    """
    Main class for users recommendations visualization.
    - Provides visual information about users in `users_dict`, their viewed items, recos and (if provided) ground truth
    - Supports saving and loading data
    - Supports easy visualization of current recommendations
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        full_recos: pd.DataFrame,
        users_dict: tp.Dict[str, tp.Any],
        ground_truth: tp.Optional[pd.DataFrame] = None,
        item_data: tp.Optional[pd.DataFrame] = None,
        n_add_random_users: int = 0,
        remove_exceeding_data: tp.Optional[bool] = True,
        auto_display: bool = True,
        convert_ids_to_int: bool = True,
        reco_cols: tp.Optional[list] = None,
    ) -> None:
        super().__init__(
            interactions=interactions,
            full_recos=full_recos,
            users_dict=users_dict,
            ground_truth=ground_truth,
            item_data=item_data,
            n_add_random_users=n_add_random_users,
            remove_exceeding_data=remove_exceeding_data,
            convert_ids_to_int=convert_ids_to_int,
        )
        self.reco_cols = reco_cols
        if auto_display:
            self.display()

    def _get_html_repr(
        self,
        items_list: np.ndarray,
        user_id: tp.Optional[int],
        model_name: tp.Optional[str],
    ) -> str:
        """
        Returns html representation of info about items in `items_list` in string format
        """
        if len(items_list) > 0:
            if isinstance(self.item_data, pd.DataFrame):
                item_df = pd.DataFrame(items_list, columns=[COLUMNS.Item])
                item_df = item_df.join(
                    self.item_data.set_index(COLUMNS.Item), on=COLUMNS.Item, how="left"
                )
                if user_id is not None and self.reco_cols is not None:
                    # user_id is provided only for display with self.reco_cols to find model results for user in reco df
                    user_recos = self.full_recos[
                        (self.full_recos[COLUMNS.User] == user_id)
                        & (self.full_recos[COLUMNS.Model] == model_name)
                    ]
                    user_recos.set_index(COLUMNS.Item, inplace=True)
                    user_recos = user_recos[self.reco_cols]
                    item_df = item_df.join(user_recos, on=COLUMNS.Item, how="left")
                    item_df_columns = PROJECT_OPTIONS.item_df_columns + self.reco_cols
                else:
                    item_df_columns = PROJECT_OPTIONS.item_df_columns
            else:
                raise TypeError("Item data was not specified")
            item_df = item_df[item_df_columns]
            item_df.rename(columns=PROJECT_OPTIONS.item_df_renaming, inplace=True)
            html_repr = (
                item_df.to_html(
                    escape=False,
                    index=False,
                    formatters=PROJECT_OPTIONS.formatters,
                    max_rows=20,
                    border=0,
                )
                .replace("<td>", '<td align="center">')
                .replace("<th>", '<th style="text-align: center; min-width: 100px;">')
            )
            return html_repr
        return "No items"

    def _get_items_tab(
        self,
        items_list: np.ndarray,
        title: str,
        user_id: tp.Optional[int],
        model_name: tp.Optional[str],
    ) -> widget_selectioncontainer.Tab:
        """
        Returns visual Tab with info about items in `items_list`
        """
        items_tab = widgets.Tab()
        items_tab.children = [
            widgets.HTML(value=self._get_html_repr(items_list, user_id, model_name))
        ]
        items_tab.set_title(index=0, title=title)
        return items_tab

    def _display_tab_for_user(
        self, user_name: str, items_type: str, model_name: str = ""
    ) -> None:
        """
        Diplays visual Tab with info about items for `user_name` depeding on `items_type` from possible
        options: `viewed`, `recos` or `ground_truth`.
        """
        user_id = self.users_dict[user_name]
        if items_type == ITEMTYPES.Viewed:
            items_list = self.get_viewed_items_for_user(user_id)
        elif items_type == ITEMTYPES.Truth:
            items_list = self.get_ground_truth_for_user(user_id)
        elif items_type == ITEMTYPES.Recos:
            items_list = self.get_recos_for_user(user_id, model_name)
        else:
            raise ValueError(f"Unknown items_type: {items_type}")
        if self.reco_cols is not None and items_type == ITEMTYPES.Recos:
            display(
                self._get_items_tab(
                    items_list, title=items_type, user_id=user_id, model_name=model_name
                )
            )
        else:
            display(
                self._get_items_tab(
                    items_list, title=items_type, user_id=None, model_name=None
                )
            )

    def _display_viewed(self, user_name: str) -> None:
        """
        Displays viewed items for `user_name`
        """
        self._display_tab_for_user(user_name, items_type=ITEMTYPES.Viewed)

    def _display_recos(self, user_name: str, model_name: str) -> None:
        """
        Displays recommended items for `user_name` from model `model_name`
        """
        self._display_tab_for_user(
            user_name, items_type=ITEMTYPES.Recos, model_name=model_name
        )

    def _display_truth(self, user_name: str) -> None:
        """
        Displays ground truth items for `user_name`
        """
        self._display_tab_for_user(user_name, items_type=ITEMTYPES.Truth)

    def _display_user_id(self, user_name: str) -> None:
        """
        Displays user_id for `user_name`
        """
        user_id = self.users_dict[user_name]
        display(widgets.HTML(value=f"User_id {user_id}"))

    def _display_model_name(self, model_name: str) -> None:
        """
        Displays user_id for `user_name`
        """
        display(widgets.HTML(value=f"Model name: {model_name}"))

    def display(self) -> None:
        """
        Displays Showcase widget with info about all users from `users_dict` providing
        visual information about viewed, recommended and (if provided) ground_truth items for each user
        """
        user = widgets.ToggleButtons(
            options=self.get_user_names(),
            description="Select user:",
            disabled=False,
            button_style="warning",
        )
        user_id_out = widgets.interactive_output(
            self._display_user_id, {"user_name": user}
        )
        viewed_out = widgets.interactive_output(
            self._display_viewed, {"user_name": user}
        )
        model = widgets.ToggleButtons(
            options=self.model_names,
            description="Select model:",
            disabled=False,
            button_style="success",
        )
        model_name_out = widgets.interactive_output(
            self._display_model_name, {"model_name": model}
        )
        recos_out = widgets.interactive_output(
            self._display_recos, {"user_name": user, "model_name": model}
        )
        if self.ground_truth is None:
            display(
                widgets.VBox(
                    [user, user_id_out, viewed_out, model, model_name_out, recos_out]
                )
            )
        else:
            truth = widgets.interactive_output(self._display_truth, {"user_name": user})
            display(
                widgets.VBox(
                    [
                        user,
                        user_id_out,
                        viewed_out,
                        model,
                        model_name_out,
                        recos_out,
                        truth,
                    ]
                )
            )

    def get_widget(self) -> widgets.Widget:
        """
        Saves Showcase widget with info about all users from `users_dict` providing
        visual information about viewed, recommended and (if provided) ground_truth items for each user
        """
        user = widgets.ToggleButtons(
            options=self.get_user_names(),
            description="Select user:",
            disabled=False,
            button_style="warning",
        )
        user_id_out = widgets.interactive_output(
            self._display_user_id, {"user_name": user}
        )
        viewed_out = widgets.interactive_output(
            self._display_viewed, {"user_name": user}
        )
        model = widgets.ToggleButtons(
            options=self.model_names,
            description="Select model:",
            disabled=False,
            button_style="success",
        )
        model_name_out = widgets.interactive_output(
            self._display_model_name, {"model_name": model}
        )
        recos_out = widgets.interactive_output(
            self._display_recos, {"user_name": user, "model_name": model}
        )
        if self.ground_truth is None:
            widget = widgets.VBox(
                [user, user_id_out, viewed_out, model, model_name_out, recos_out]
            )
        else:
            truth = widgets.interactive_output(self._display_truth, {"user_name": user})
            widget = widgets.VBox(
                [
                    user,
                    user_id_out,
                    viewed_out,
                    model,
                    model_name_out,
                    recos_out,
                    truth,
                ]
            )
        
        return widget
    

    def _make_name_from_recos_date(self) -> str:
        """
        Generate name automatically if 'date' is in `full_recos`
        """
        if COLUMNS.Date in self.full_recos.columns:
            reco_date = str(self.full_recos[COLUMNS.Date].values[0])
            name = f"real_recos_{reco_date}"
            print(f"Saving with name: {name}")
            return name
        raise ValueError("Name not specified and `date` not in `full_recos.columns`")
 