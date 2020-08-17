"""Pokémans data now!!!"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Globals
SEED = 314
poketype_colors = {"Water": "#6890F0",
                   "Fire": "#F08030",
                   "Grass": "#78C850",
                   "Electric": "#F8D030",
                   "Ice": "#98D8D8",
                   "Psychic": "#F85888",
                   "Dragon": "#7038F8",
                   "Dark": "#705848",
                   "Fairy": "#EE99AC",
                   "Normal": "#A8A878",
                   "Fighting": "#C03028",
                   "Flying": "#A890F0",
                   "Poison": "#A040A0",
                   "Ground": "#E0C068",
                   "Rock": "#B8A038",
                   "Bug": "#A8B820",
                   "Ghost": "#705898",
                   "Steel": "#B8B8D0",
                   "Unknown": "#68A090"}


def cleaned_pokemon(path) -> pd.DataFrame:
    pokemon = pd.read_csv(path)
    abilities_temp_name = "abilities_tmp"

    # Classification was totally not my misspelling
    pokemon.rename(columns={"classfication": "category",
                            "abilities": abilities_temp_name},
                   inplace=True)

    # Melt types1/2 into a tidy column
    pokemon = pokemon.melt(id_vars=pokemon.columns.difference(["type1",
                                                               "type2"]),
                           var_name="type_order",
                           value_name="type")

    # Clean up array specifiers in each ability list
    # Removes [, ], and '
    pokemon.abilities_tmp = pokemon.abilities_tmp.str.replace("\\[|'|\\]", "")

    # Next, let's split the list into separate columns which I'll merge
    # back into the Pokémon DataFrame. I'll also store a copy of the column
    # names without the new merged column names to use for melting.
    poke_orig_cols = pokemon.columns.copy()
    abilities = pokemon.abilities_tmp.str.split(", ", expand=True)
    pokemon = pokemon.merge(abilities, how="outer", left_index=True,
                            right_index=True)

    # Now let's clean up by melting/gathering the abilities.
    # Melting produces a new DataFrame with duplicated entries for the
    # Pokémon with more than one ability.
    pokemon = pokemon.melt(id_vars=poke_orig_cols,
                           var_name="type_order",
                           value_name="abilities")

    # Finally, drop the old columns as well as nulls introduced
    # from the split. Check out the R port of the code for an explanation.
    # cols_to_drop = [abilities_temp_name, drop_temp_name]
    pokemon.drop(columns=abilities_temp_name, inplace=True)
    pokemon.dropna(subset=["abilities", "type"], inplace=True)

    # What's next? Let's fix up variable types!
    cat_cols = ["category", "generation", "abilities", "type",
                "base_egg_steps", "base_happiness", "is_legendary",
                "is_mythical", "is_mega"]

    for col in cat_cols:
        pokemon[col] = pokemon[col].astype("category")

    pokemon.pokedex_number = pokemon.pokedex_number.astype(int)
    pokemon.name = pokemon.name.astype("string")

    # Finally, let's sort and return the DataFrame
    pokemon.sort_values(by="pokedex_number", inplace=True)
    pokemon.reset_index(drop=True, inplace=True)
    return pokemon


def flatten_pokemon(pokemon: pd.DataFrame) -> pd.DataFrame:
    pokedex = pokemon.pokedex_number.unique()
    newpoke = pd.get_dummies(pokemon, columns=["type", "abilities"])

    # Can't figure out a smart way to apply/map
    for pokenum in pokedex:
        types = pokemon.loc[pokemon.pokedex_number == pokemon, "type"].unique()

        newpoke.loc[newpoke.pokedex_number == pokenum,]

