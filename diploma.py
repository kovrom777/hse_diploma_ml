# -*- coding: utf-8 -*-
"""diploma.ipynb
# Imports
"""

!pip install LightFM

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split

"""# Help functions"""

def precision_recall_at_k(actual, predicted, k=10):
    actual_set = set(actual[:k])
    predicted_set = set(predicted[:k])
    intersection = len(actual_set & predicted_set)

    precision = intersection / k if k else 0
    recall = intersection / len(actual_set) if actual_set else 0

    return precision, recall

"""# Datasets preparation"""

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

users_file = "users.json"
history_file = "history.json"
products_file = "products.json"

users_data = load_json(users_file)["data"]
history_data = load_json(history_file)["data"]
products_data = load_json(products_file)["data"]

users_list = []
for user_id, user_info in users_data.items():
    if "__collections__" in user_info and "adresses" in user_info["__collections__"]:
        for address_id, address_info in user_info["__collections__"]["adresses"].items():
            users_list.append({
                "user_id": user_id,
                "name": user_info.get("name", ""),
                "street": address_info.get("street", ""),
                "house": address_info.get("house", ""),
                "entrance": address_info.get("buildingEntrance", ""),
                "apartment": address_info.get("doorNumber", ""),
            })

users_df = pd.DataFrame(users_list)

users_df.head()

products_list = []
for city_id, city_info in products_data.items():
    if "__collections__" in city_info and "products" in city_info["__collections__"]:
        for product_id, product_info in city_info["__collections__"]["products"].items():
            products_list.append({
                "product_id": product_id,
                "name": product_info.get("name", ""),
                "price": product_info.get("price", 0),
                "category": product_info["categoryRef"]["__ref__"].split("/")[-1] if "categoryRef" in product_info else None,
                "discount": product_info.get("allowDiscount", False),
                "image": product_info.get("iconImage", ""),
                "description": product_info.get("description", "")
            })

products_df = pd.DataFrame(products_list)

products_df.head()

orders_list = []
for order_id, order_info in history_data.items():
    user_ref = None
    if isinstance(order_info.get("userRef"), dict):
        user_ref = order_info["userRef"].get("__ref__", "").split("/")[-1]

    if "__collections__" in order_info and "products" in order_info["__collections__"]:
        for product_id, product_info in order_info["__collections__"]["products"].items():
            product_ref = None
            if isinstance(product_info.get("productRef"), dict):
                product_ref = product_info["productRef"].get("__ref__", "").split("/")[-1]

            orders_list.append({
                "order_id": order_id,
                "user_id": user_ref,
                "product_id": product_ref,
                "count": product_info.get("count", 1),
                "price": order_info.get("price", 0),
                "address": order_info.get("adress", ""),
                "order_date": order_info.get("orderDate", {}).get("__time__", None)
            })

orders_df = pd.DataFrame(orders_list)

orders_df.head()

merged_df = orders_df.merge(users_df, on="user_id", how="left").merge(products_df, on="product_id", how="left")

merged_df["family_id"] = (
    merged_df["street"] + "_" + merged_df["house"].astype(str) + "_" +
    merged_df["entrance"].astype(str) + "_" + merged_df["apartment"].astype(str)
)

merged_df.head()

merged_df.info()

merged_df.isna().sum()

merged_df.shape

merged_df.columns

final = merged_df.copy()

final = final.drop(columns={'image', 'street', 'house', 'entrance', 'apartment', 'address'})

final = final.rename(columns={
    'price_x': 'order_price',
    'name_x': 'client_name',
    'name_y': 'product_name',
    'price_y': 'product_price'
    })

final.head()

final.to_csv('final.csv', index=False)

"""Categories id\

0f7kp5wcm73AVoS7u71s - Акции\
0kqAvq7OTumJ85PyW8nr - Закуски и салаты\
58QZ5QsfQy1K8PT0jbks - сеты\
BFvVAGV9rV6Fkp1DZTwm - соуса\
NOn81HMZPYytNvEou0Jr - напитки и десерты\
Uv2uq1jnf1NC7JI13V56 - запеченые роллы\
gCzCO39jbY1RRyfti8yE - пицца\
v0ufIpn2ogrwKL6hvhgj - роллы

## Removing Nans
"""

final.isna().sum()

"""# Working with merged dataset"""

merged = pd.read_csv('final.csv')
merged.head()

merged.isna().sum()

merged_copy = merged.copy()
merged_copy = merged_copy.dropna(subset=['product_name', 'family_id'])
merged_copy.isna().sum()

merged_copy[:3]

merged_copy.to_csv('final_without_nans.csv')

"""## Подсчет "холодных" и "горячих" пользователей"""

merged_without_nans = pd.read_csv('final_without_nans.csv')

num_families = merged_without_nans["family_id"].nunique()

num_dishes = merged_without_nans["product_id"].nunique()

avg_orders_per_family = merged_without_nans.groupby("family_id")["order_id"].nunique().mean()

orders_per_family = merged_without_nans.groupby("family_id")["order_id"].nunique()

cold_users_ratio = (orders_per_family <= 3).sum() / num_families

{
    "Всего заказов": len(merged_without_nans),
    "Уникальных семей": num_families,
    "Уникальных блюд": num_dishes,
    "Среднее число заказов на семью": avg_orders_per_family,
    "Процент холодных пользователей (≤3 заказа)": cold_users_ratio * 100
}

"""# Building lightFM model"""

merged_without_nans = pd.read_csv('final_without_nans.csv')

excluded_categories = [
    "0kqAvq7OTumJ85PyW8nr",  # Закуски и салаты
    "58QZ5QsfQy1K8PT0jbks",  # Сеты
    "BFvVAGV9rV6Fkp1DZTwm",  # Соусы
    "NOn81HMZPYytNvEou0Jr"   # Напитки и десерты
]
merged_without_nans["ingredients_list"] = merged_without_nans.apply(
    lambda row: str(row["description"]).split(", ") if row["category"] not in excluded_categories else "", axis=1)

df = merged_without_nans.copy()

users = df['family_id'].unique()
items = df['product_id'].unique()

df["ingredients_list"] = df["ingredients_list"].apply(lambda lst: [i.strip() for i in lst if i.strip() != ""])

all_ingredients_tags = []
for i in range(len(df)):
  row_ingredients = df.loc[i, "ingredients_list"]

  if isinstance(row_ingredients, list):
      tags = []
      for ingredient in row_ingredients:
          if isinstance(ingredient, str) and ingredient.strip():
              tags.append(f"ingredient:{ingredient.strip()}")
          else:
            all_ingredients_tags.append(f"ingredient:None")
      all_ingredients_tags.append(tags)
  else:
    print(row_ingredients)
    all_ingredients_tags.append(f"ingredient:None")

df["ingredient_tags"] = all_ingredients_tags

s = set()
for tag in all_ingredients_tags:
  for elem in tag:
    s.add(elem)

s.add('ingredient:None')

dataset = Dataset()
dataset.fit(users=users, items=items, item_features=s)

(interactions, _) = dataset.build_interactions(
    (row["family_id"], row["product_id"]) for _, row in df.iterrows()
)

item_features_data = []
for product_id in items:
    sub = df[df["product_id"] == product_id]
    tags = set()
    for tag_list in sub["ingredient_tags"]:
        tags.update(tag_list)

    if not tags:
        tags.add("ingredient:None")

    item_features_data.append((product_id, list(tags)))

item_features_matrix = dataset.build_item_features(item_features_data)

train, test = random_train_test_split(
    interactions,
    test_percentage=0.2,
    random_state=42
)

model = LightFM(loss="warp")
model.fit(train, item_features=item_features_matrix, epochs=10, num_threads=4)

precision = precision_at_k(
    model, test, k=10,
    item_features=item_features_matrix
).mean()
print(f"✅ Precision@10 (только по ингредиентам): {precision:.4f}")

"""генерируется топ-10 рекомендаций для семьи\
0.25 неплохой результат при условии, что в датасете всего 94 блюда, некоторые из которых исключены из финальных фич по обучения (соусы, напитки)\
так же почти 76% клиентов - холодные

## LightFM with cats
"""

cat_df = df.copy()

cat_df["category_tag"] = cat_df["category"].astype(str).str.strip().apply(lambda c: f"category:{c}")
cat_df.head(1)

cat_df["item_tags"] = cat_df.apply(lambda row: row["ingredient_tags"] + [row["category_tag"]], axis=1)

all_tags = set()
for tag_list in cat_df["item_tags"]:
    all_tags.update(tag_list)

cat_dataset = Dataset()
cat_dataset.fit(users=users, items=items, item_features=all_tags)

(interactions, _) = cat_dataset.build_interactions(
    (row["family_id"], row["product_id"]) for _, row in df.iterrows()
)

item_features_data_cat = []
for product_id in items:
    group = cat_df[df["product_id"] == product_id]
    tags = set()
    for tag_list in group["item_tags"]:
        tags.update(tag_list)
    item_features_data_cat.append((product_id, list(tags)))

item_features_cat_matrix = cat_dataset.build_item_features(item_features_data_cat)

item_features_cat_matrix = cat_dataset.build_item_features(item_features_data_cat)

cat_model = LightFM(loss="warp")
cat_model.fit(train, item_features=item_features_cat_matrix, epochs=10, num_threads=4)

precision = precision_at_k(
    cat_model, test, k=10,
    item_features=item_features_cat_matrix
).mean()
print(f"✅ Precision@10 (ingr + cats): {precision:.4f}")

"""категории не улучшили метрику

## LightFM with converting sets
"""

set_category_id = '58QZ5QsfQy1K8PT0jbks'

set_df = df[df["category"] == set_category_id].copy()
product_ingredients_map = df.set_index("product_name")["ingredients_list"].to_dict()

def extract_ingredients_from_set_description(description):
    try:
        roll_names = [r.strip() for r in description.split(",") if r.strip()]
        ingredients = []
        for roll in roll_names:
            if roll in product_ingredients_map:
                ingredients += product_ingredients_map[roll]
        return list(set(ingredients))  # убираем дубли
    except Exception:
        return []

set_df["ingredients_list"] = set_df["description"].apply(extract_ingredients_from_set_description)

needed_set = set_df[["product_id", "ingredients_list"]]
needed_set.head()

merged_df = df.merge(needed_set, on="product_id", how="left")

merged_df["ingredients_list"] = merged_df.apply(
    lambda row: list(set(
        (row["ingredients_list_x"] if isinstance(row["ingredients_list_x"], list) else []) +
        (row["ingredients_list_y"] if isinstance(row["ingredients_list_y"], list) else [])
    )),
    axis=1
)

merged_df.drop(columns=["ingredients_list_x", "ingredients_list_y"], inplace=True)
merged_df.head(1)

all_ingredients_tags = []
for i in range(len(merged_df)):
  row_ingredients = merged_df.loc[i, "ingredients_list"]

  if isinstance(row_ingredients, list):
      tags = []
      for ingredient in row_ingredients:
          if isinstance(ingredient, str) and ingredient.strip():
              tags.append(f"ingredient:{ingredient.strip()}")
      all_ingredients_tags.append(tags)
  else:
      all_ingredients_tags.append([])  # пустой список, если нет данных

merged_df["ingredient_tags"] = all_ingredients_tags

s = set()
for tag in all_ingredients_tags:
  for elem in tag:
    s.add(elem)

sets_dataset = Dataset()
sets_dataset.fit(users=users, items=items, item_features=s)

(interactions, _) = sets_dataset.build_interactions(
    (row["family_id"], row["product_id"]) for _, row in df.iterrows()
)

# Создание item_features
item_features_sets_data = []
for product_id, group in df.groupby("product_id"):
    tags = set()
    for tag_list in group["ingredient_tags"]:
        tags.update(tag_list)
    item_features_sets_data.append((product_id, list(tags)))

item_features_sets_matrix = sets_dataset.build_item_features(item_features_sets_data)

sets_model = LightFM(loss="warp")
sets_model.fit(train, item_features=item_features_sets_matrix, epochs=10, num_threads=4)

precision = precision_at_k(
    sets_model, test, k=10,
    item_features=item_features_sets_matrix
).mean()
print(f"✅ Precision@10 (ingr+sets): {precision:.4f}")

"""## LightFM with ingr + cats + sets"""

full_df = merged_df.copy()
full_df.head(1)

full_df["category_tag"] = full_df["category"].apply(lambda c: f"category:{c}")

full_df["item_tags"] = full_df.apply(lambda row: row["ingredient_tags"] + [row["category_tag"]], axis=1)

all_tags = set()
for tag_list in full_df["item_tags"]:
    all_tags.update(tag_list)

cat_dataset = Dataset()
cat_dataset.fit(users=users, items=items, item_features=all_tags)

(interactions, _) = cat_dataset.build_interactions(
    (row["family_id"], row["product_id"]) for _, row in df.iterrows()
)

item_features_data_cat = []
for product_id in items:
    group = cat_df[df["product_id"] == product_id]
    tags = set()
    for tag_list in group["item_tags"]:
        tags.update(tag_list)
    item_features_data_cat.append((product_id, list(tags)))

item_features_cat_matrix = cat_dataset.build_item_features(item_features_data_cat)

full_dataset = Dataset()
full_dataset.fit(users=users, items=items, item_features=all_tags)

(interactions, _) = full_dataset.build_interactions(
    (row["family_id"], row["product_id"]) for _, row in df.iterrows()
)

item_features_data_full = []
for product_id in items:
    group = cat_df[df["product_id"] == product_id]
    tags = set()
    for tag_list in group["item_tags"]:
        tags.update(tag_list)
    item_features_data_full.append((product_id, list(tags)))

item_features_full_matrix = full_dataset.build_item_features(item_features_data_full)

item_features_full_matrix = cat_dataset.build_item_features(item_features_data_full)

full_model = LightFM(loss="warp")
full_model.fit(train, item_features=item_features_full_matrix, epochs=10, num_threads=4)

precision = precision_at_k(
    full_model, test, k=10,
    item_features=item_features_full_matrix
).mean()
print(f"✅ Precision@10 (ingr + cats + sets): {precision:.4f}")

"""Precision@10 (ingr): 0.1564\
Precision@10 (ingr + cats): 0.1557\
Precision@10 (ingr + sets): 0.1569\
Precision@10 (ingr + cats + sets): 0.1560

## LightFM predict
"""

merged_without_nans = pd.read_csv('final_without_nans.csv')

df = merged_without_nans.copy()

users = df['family_id'].unique()
items = df['product_id'].unique()

excluded_categories = [
  "0kqAvq7OTumJ85PyW8nr",  # Закуски и салаты
  "58QZ5QsfQy1K8PT0jbks",  # Сеты
  "BFvVAGV9rV6Fkp1DZTwm",  # Соусы
  "NOn81HMZPYytNvEou0Jr"   # Напитки и десерты
]

df["ingredients_list"] = df.apply(
  lambda row: str(row["description"]).split(", ") if row["category"] not in excluded_categories else [],
  axis=1
)

    # Clean ingredients
df["ingredients_list"] = df["ingredients_list"].apply(
  lambda lst: [i.strip() for i in lst if isinstance(i, str) and i.strip() != ""]
)

    # Create ingredient tags
df["ingredient_tags"] = df["ingredients_list"].apply(
  lambda ingredients: [f"ingredient:{ingredient}" for ingredient in ingredients]
)

users = df['family_id'].unique()
items = df['product_id'].unique()

dataset = Dataset()
dataset.fit(
    users=users,
    items=items,
    item_features=set(tag for sublist in df["ingredient_tags"] for tag in sublist)
)

(interactions, weights) = dataset.build_interactions(
    (row["family_id"], row["product_id"]) for _, row in df.iterrows()
)

item_features_data = []
for product_id in items:
    product_data = df[df["product_id"] == product_id]
    tags = set(tag for sublist in product_data["ingredient_tags"] for tag in sublist)
    item_features_data.append((product_id, list(tags)))

item_features = dataset.build_item_features(item_features_data)

train, test = random_train_test_split(
    interactions,
    test_percentage=0.2,
    random_state=42
)

model = LightFM(loss="warp", random_state=42)
model.fit(
    train,
    item_features=item_features,
    epochs=10,
    num_threads=4
)

precision = precision_at_k(
    model, test, k=10,
    item_features=item_features
).mean()
print(f"Precision@10: {precision:.4f}")

def get_recommendations(family_id, model, dataset, df, n_recs=10):
    all_item_ids = list(df['product_id'].unique())
    known_items = set(df[df['family_id'] == family_id]['product_id'])
    user_id_map, _, item_id_map, _ = dataset.mapping()
    if family_id not in user_id_map:
        print(f"Family {family_id} not found in the model. Returning popular items.")
        popular_items = df['product_id'].value_counts().head(n_recs).index.tolist()
        return df[df['product_id'].isin(popular_items)][['product_id', 'product_name', 'description']].drop_duplicates()

    valid_items = [item for item in all_item_ids if item in item_id_map]
    if not valid_items:
        raise ValueError("No valid items found in the model mapping")

    scores = model.predict(
        user_ids=user_id_map[family_id],
        item_ids=[item_id_map[item] for item in valid_items],
        item_features=item_features_matrix
    )

    recommendations = pd.DataFrame({
        'product_id': valid_items,
        'score': scores
    })

    recommendations = recommendations[~recommendations['product_id'].isin(known_items)]

    top_recs = recommendations.sort_values('score', ascending=False).head(n_recs)
    top_recs = top_recs.merge(
        df[['product_id', 'product_name', 'description']].drop_duplicates(),
        on='product_id',
        how='left'
    )

    return top_recs

sample_family = df['family_id'].iloc[321]
sample_family

recommendations = get_recommendations(sample_family, model, dataset, df)

if recommendations is not None:
    print("Recommendations for family:", sample_family)
    print(recommendations)

"""# Model export"""

user_id_map, _, item_id_map, reverse_item_id_map = dataset.mapping()

import pickle
import scipy.sparse

with open("lightfm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("mappings.pkl", "wb") as f:
    pickle.dump({
        "user_map": user_id_map,
        "item_map": item_id_map,
        "reverse_item_map": reverse_item_id_map
    }, f)

scipy.sparse.save_npz("item_features_matrix.npz", item_features_matrix)

"""# Model import"""

import pickle
import numpy as np
import scipy.sparse
from lightfm import LightFM

with open("lightfm_model.pkl", "rb") as f:
    new_model = pickle.load(f)

with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

item_features_matrix = scipy.sparse.load_npz("item_features_matrix.npz")

user_id_map = mappings["user_map"]
item_id_map = mappings["item_map"]
reverse_item_id_map = mappings["reverse_item_map"]

def get_recommendations_for_saved(
    family_id,
    model,
    user_id_map,
    item_id_map,
    df,
    n_recs=10
):
    all_item_ids = list(df['product_id'].unique())
    known_items = set(df[df['family_id'] == family_id]['product_id'])

    if family_id not in user_id_map:
        print(f"Family {family_id} not found in the model. Returning popular items.")
        popular_items = df['product_id'].value_counts().head(n_recs).index.tolist()
        return df[df['product_id'].isin(popular_items)][['product_id', 'product_name', 'description']].drop_duplicates()

    valid_items = [item for item in all_item_ids if item in item_id_map]
    if not valid_items:
        raise ValueError("No valid items found in the model mapping")

    scores = model.predict(
        user_ids=user_id_map[family_id],
        item_ids=[item_id_map[item] for item in valid_items],
        item_features=item_features_matrix
    )

    recommendations = pd.DataFrame({
        'product_id': valid_items,
        'score': scores
    })

    recommendations = recommendations[~recommendations['product_id'].isin(known_items)]

    top_recs = recommendations.sort_values('score', ascending=False).head(n_recs)

    top_recs = top_recs.merge(
        df[['product_id', 'product_name', 'description']].drop_duplicates(),
        on='product_id',
        how='left'
    )

    return top_recs

sample_family = df['family_id'].iloc[123]
sample_family

recommendations = get_recommendations_for_saved(
    sample_family,
    new_model,
    user_id_map,
    item_id_map,
    df
)

if recommendations is not None:
    print("Recommendations for family:", sample_family)
    print(recommendations)
