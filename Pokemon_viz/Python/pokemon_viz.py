"""Pokémans data now!!!"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cleaned_pokemon(path) -> pd.DataFrame:
    pokemon = pd.read_csv(path)
    abilities_temp_name = "abilities_tmp"

    # Classification was totally not my misspelling
    pokemon.rename(columns={"classfication": "category",
                            "abilities": abilities_temp_name},
                   inplace=True)

    # Clean up array specifiers in each ability list
    # Removes [, ], and '
    pokemon.abilities_tmp = pokemon.abilities_tmp.str.replace("\\[|'|\\]", "")

    # Next, let's split the list into separate columns which I'll merge
    # back into the Pokémon DataFrame
    abilities = pokemon.abilities_tmp.str.split(", ", expand=True)
    pokemon = pokemon.merge(abilities, how="outer", left_index=True,
                            right_index=True)

    # Now let's clean up by melting/gathering the abilities as well as
    # selecting the new column as a Series.
    # Also! Merge the result back into the DataFrame as the new abilities col.
    # I'm sure I can figure out how to clean up this process, but I'm fine
    # with my code for now.
    abilities_melted = pokemon.melt(value_vars=abilities.columns,
                                    value_name="abilities").abilities
    pokemon = pokemon.merge(abilities_melted, how="outer", left_index=True,
                            right_index=True)

    # Finally, drop the old columns as well as nulls introduced
    # from the split. Check out the R port of the code for an explanation.
    cols_to_drop = list(abilities.columns).append(abilities_temp_name)
    pokemon.drop(columns=cols_to_drop, inplace=True)
    pokemon.dropna(subset=["abilities"], inplace=True)

    return pokemon
