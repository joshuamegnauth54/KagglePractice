"""Pokémans data now!!!"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Globals
SEED = 314

def cleaned_pokemon(path) -> pd.DataFrame:
    pokemon = pd.read_csv(path)
    abilities_temp_name = "abilities_tmp"
    drop_temp_name = "to_drop"

    # Classification was totally not my misspelling
    pokemon.rename(columns={"classfication": "category",
                            "abilities": abilities_temp_name},
                   inplace=True)

    # Melt types1/2 into a tidy column
    pokemon = pokemon.melt(id_vars=pokemon.columns.difference(["type1",
                                                               "type2"]),
                           var_name=drop_temp_name,
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
                           var_name=drop_temp_name,
                           value_name="abilities")

    # Finally, drop the old columns as well as nulls introduced
    # from the split. Check out the R port of the code for an explanation.
    cols_to_drop = [abilities_temp_name, drop_temp_name]
    pokemon.drop(columns=cols_to_drop, inplace=True)
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
