import streamlit as st 
import pandas as pd 
import difflib
import re 

def remove_non_numeric_prefix(ref):
    """Supprime les caractères non numériques au début de la chaine."""
    if pd.isna(ref):
        return ''
    return re.sub(r'^\D+', '', str(ref))

def has_non_numeric_leading(ref):
    """Vérifie si la référence commence par des caractères non numériques."""
    if pd.isna(ref):    
        return False
    return bool(re.match(r'^\D+', str(ref)))

def remove_leading_zeros(ref):
    """
    Supprime les zéros non significatifs en tête d'une chaîne de référence.
    """
    if pd.isna(ref):
        return ''
    return ref.lstrip('0')

def has_leading_zeros(ref):
    """
    Vérifie si la chaîne de référence contient des zéros non significatifs au début
    """
    if pd.isna(ref):
        return False
    return bool(re.search(r'^0+\d', ref))

def remove_non_alphanumeric(ref):
    """
    Supprime tous les caractères non alphanumériques de la chaîne.
    """
    if pd.isna(ref):
        return ''
    return re.sub(r'[^a-zA-Z0-9]', '', ref)

def has_special_characters(ref):
    
    if pd.isna(ref):
        return ''
    return bool(re.search(r'[^a-zA-Z0-9]', ref))

def is_single_replacement(s1, s2):
    """
    Vérifie si deux chaînes de caractères diffèrent par un seul remplacement de chiffre par lettre ou vice versa
    """
    if len(s1) != len(s2):
        return False, None, None
    differences = 0
    replaced_char_s1 = replaced_char_s2 = None
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if (c1.isdigit() and not c2.isdigit()) or (not c1.isdigit() and c2.isdigit()):
                differences += 1
                replaced_char_s1, replaced_char_s2 = c1, c2
                if differences > 1:
                    return False, None, None
            else:
                return False, None, None
    if differences == 1:
        return True, replaced_char_s1, replaced_char_s2
    return False, None, None

def process_dataframe(df):
    """Traite le DataFrame en ajoutant les colonnes normalisées."""
    try:
        df['Normalized_Ref'] = df['Référence'].apply(remove_non_numeric_prefix)
        df['Has_Non_Numeric_Leading'] = df['Référence'].apply(has_non_numeric_leading)
        df['Normalized_Ref_zeros'] = df['Normalized_Ref'].apply(remove_leading_zeros)
        df['Has_Leading_Zeros'] = df['Normalized_Ref'].apply(has_leading_zeros)
        df['Normalized_Ref_alpha'] = df['Normalized_Ref_zeros'].apply(remove_non_alphanumeric)
        df['Has_Leading_Alpha'] = df['Normalized_Ref_zeros'].apply(has_special_characters)

        for doublon_value, group in df.groupby('Doublon'):
            refs = group['Référence'].tolist()
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    result, char1, char2 = is_single_replacement(refs[i], refs[j])
                    if result:
                        if char1.isdigit() and not char2.isdigit():
                            numeric, alpha = refs[i], refs[j]
                        else:
                            numeric, alpha = refs[j], refs[i]
                        data_frame.loc[(df['Normalized_Ref_spe_charac'] == alpha) & (df['Doublon'] == doublon_value), 'Normalized_Ref_spe_charac'] = numeric
                        st.write(f"\nTraitement du groupe '{doublon_value}':")
                        st.write(f"Remplacement effectué : '{alpha}' remplacé par '{numeric}'")
        
        # Sélection des colonnes spécifiques
        columns_to_display = ['Référence', 'Doublon', 'Normalized_Ref', 'Has_Non_Numeric_Leading', 'Normalized_Ref_zeros', 'Has_Leading_Zeros', 'Normalized_Ref_alpha', 'Has_Leading_Alpha']
        df_display = df[columns_to_display]
        
        return df_display
    except KeyError:
        st.error("La colonne 'Référence' n'a pas été trouvée dans le fichier.")
        return None

def traiter_fichier(uploaded_file):
    """Traite le fichier Excel uploadé."""
    if uploaded_file is None:
        st.warning("Veuillez télécharger un fichier Excel.")
        return
    
    try:
        st.write(f"Fichier reçu: {uploaded_file.name}")
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("Aperçu du fichier original (colonnes sélectionnées) :")
        st.dataframe(df.head())
        
        # Traitement du DataFrame
        processed_df = process_dataframe(df)
        if processed_df is not None:
            st.write("Aperçu du fichier traité :")
            st.dataframe(processed_df.head())
            st.success("Fichier traité avec succès !")
            
            return processed_df
            
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
        return None

def main():
    st.title("Outil de traitement du fichier")
    uploaded_file = st.file_uploader("Déposez votre fichier ici", type=["xlsx"])
    
    if uploaded_file:
        processed_df = traiter_fichier(uploaded_file)
        if processed_df is not None:
            # Vous pouvez ajouter ici d'autres traitements sur processed_df
            pass

if __name__ == "__main__":
    main()

"""
import streamlit as st 
import pandas as pd 
import difflib
import re 


def traiter_fichier(uploaded_file):
    if uploaded_file is not None : 
        st.write(f"Fichier reçu: {uploaded_file.name}")

    #try: 
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("Aperçu du fichier")
        st.dataframe(df.head())
        #st.dataframe(df, use_container_width=True)
      
        st.success("Fichier traité avec succès !")
    
    except Exception as e:
        e = "Veuilliez télécharger le fichier Excel"
        st.error(f"Erreur lors de la lecture du fichier : {e}")
    
st.title("Outil de traitement du fichier")
uploaded_file = st.file_uploader("Déposez votre fichier ici", type =["xlsx"])
traiter_fichier(uploaded_file)


# Supprime les caractères non numériques au début de la chaine
def remove_non_numeric_prefix(ref):
    if pd.isna(ref):
        return ''
    return re.sub(r'^\D+', '', ref)

def has_non_numeric_leading(ref):
    if pd.isna(ref):    
        return ''
    return bool(re.match(r'^\D+', ref))

data_frame = pd.DataFrame(df)
data_frame['Normalized_Ref'] = data_frame['Référence'].apply(remove_non_numeric_prefix)
data_frame['Has_Non_Numeric_Leading'] = data_frame['Référence'].apply(has_non_numeric_leading)
"""
"""
def remove_leading_zeros(ref):
    if pd.isna(ref):
        return ''
    return ref.lstrip('0')

def has_leading_zeros(ref):
    if pd.isna(ref):
        return False
    return bool(re.search(r'^0+\d', ref))

def remove_leading_zeros(ref):
    if pd.isna(ref):
        return ''
    return ref.lstrip('0')

def has_leading_zeros(ref):
    if pd.isna(ref):
        return False
    return bool(re.search(r'^0+\d', ref))

def remove_non_alphanumeric(ref):
    if pd.isna(ref):
        return ''
    return re.sub(r'[^a-zA-Z0-9]', '', ref)

def has_special_characters(ref):
    if pd.isna(ref):
        return ''
    return bool(re.search(r'[^a-zA-Z0-9]', ref))

def is_single_replacement(s1, s2):
    if len(s1) != len(s2):
        return False, None, None
    differences = 0
    replaced_char_s1 = replaced_char_s2 = None
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if (c1.isdigit() and not c2.isdigit()) or (not c1.isdigit() and c2.isdigit()):
                differences += 1
                replaced_char_s1, replaced_char_s2 = c1, c2
                if differences > 1:
                    return False, None, None
            else:
                return False, None, None
    if differences == 1:
        return True, replaced_char_s1, replaced_char_s2
    return False, None, None

def is_single_addition_or_deletion(s1, s2):
    if len(s1) == len(s2) + 1:
        for i in range(len(s2) + 1):
            if s1[:i] + s1[i+1:] == s2:
                if i == len(s2) and s1[-1].isalpha():
                    continue
                return True
        return False
    elif len(s2) == len(s1) + 1:
        for i in range(len(s1) + 1):
            if s2[:i] + s2[i+1:] == s1:
                if i == len(s1) and s2[-1].isalpha():
                    continue
                return True
        return False
    else:
        return False

def update_refs_within_group(group):
    new_rows = []
    rows_to_remove = set()
    affected_rows = []

    for i, row_i in group.iterrows():
        for j, row_j in group.iterrows():
            if i != j and is_single_addition_or_deletion(row_i['Normalized_Ref'], row_j['Normalized_Ref']):
                if len(row_i['Normalized_Ref']) < len(row_j['Normalized_Ref']):
                    new_row = row_i.copy()
                    new_row['Normalized_Ref'] = row_j['Normalized_Ref']
                    new_rows.append(new_row)
                    rows_to_remove.add(i)
                    affected_rows.append((row_i['Normalized_Ref'], row_j['Normalized_Ref']))

    group = group.drop(index=list(rows_to_remove))

    if new_rows:
        group = pd.concat([group, pd.DataFrame(new_rows)], ignore_index=True)

    return group, affected_rows

def has_missing_character(ref, group_refs):
    for other_ref in group_refs:
        if other_ref == ref:
            continue  
        if ref in other_ref:
            extra_part = other_ref.replace(ref, '', 1)
            if not (extra_part.isalpha() and other_ref.endswith(extra_part)):
                return True  
        for i in range(len(other_ref)):
            modified_ref = other_ref[:i] + other_ref[i+1:]
            if ref == modified_ref and not (i == len(ref) and other_ref[i].isalpha()):
                return True  

    return False  

def update_refs_within_group(group):
    # Exemple de logique pour mettre à jour les références dans le groupe
    affected_rows = []
    for index, row in group.iterrows():
        new_ref = row['Normalized_Ref'] + '_updated'  # Exemple de mise à jour
        affected_rows.append((row['Normalized_Ref'], new_ref))
        group.at[index, 'Normalized_Ref'] = new_ref
    return group, affected_rows

# Ton code existant
df['Has_Missing_Char'] = False

for doublon_value, group in df.groupby('Doublon'):
    group_refs = group['Normalized_Ref'].tolist()
    for index, row in group.iterrows():
        df.loc[index, 'Has_Missing_Char'] = has_missing_character(row['Normalized_Ref'], group_refs)

all_new_rows = pd.DataFrame()
affected_groups = []

for doublon, group in df.groupby('Doublon', sort=False):
    updated_group, affected_rows = update_refs_within_group(group)
    all_new_rows = pd.concat([all_new_rows, updated_group], ignore_index=True)
    if affected_rows:
        affected_groups.append((doublon, affected_rows))

df = all_new_rows

for doublon, affected_rows in affected_groups:
    print(f"Traitement du groupe '{doublon}':")
    for old_ref, new_ref in affected_rows:
        print(f"Remplacement effectué : '{old_ref}' remplacé par '{new_ref}'")
    print()

print("DataFrame après les mises à jour :")
print(df)
"""