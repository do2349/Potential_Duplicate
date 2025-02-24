import streamlit as st
import pandas as pd
import difflib
import re
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Traitement des références", layout="wide")

# Caractères alphabétiques d'en tête 
def remove_non_numeric_prefix(ref):
    if pd.isna(ref):
        return ''
    return re.sub(r'^\D+', '', str(ref))

def has_non_numeric_leading(ref):
    if pd.isna(ref):    
        return False
    return bool(re.match(r'^\D+', str(ref)))

# Enlever les zéros non significatifs 
def remove_leading_zeros(ref):
    if pd.isna(ref):
        return ''
    return str(ref).lstrip('0')  # Conversion explicite en string

def has_leading_zeros(ref):
    if pd.isna(ref):
        return False
    return bool(re.search(r'^0+\d', str(ref)))  # Conversion explicite en string

# Traitement des caractères spéciaux 
def remove_non_alphanumeric(ref):
    if pd.isna(ref):
        return ''
    return re.sub(r'[^a-zA-Z0-9]', '', str(ref))  # Conversion explicite en string

def has_special_characters(ref):
    if pd.isna(ref):
        return False
    return bool(re.search(r'[^a-zA-Z0-9]', str(ref)))  # Conversion explicite en string

# Remplacement d'un chiffre par une lettre 
def is_single_replacement(s1, s2):
    """
    Vérifie si deux chaînes de caractères diffèrent par un seul remplacement de chiffre par lettre ou vice versa
    """
    # Conversion explicite en string
    s1, s2 = str(s1), str(s2)
    
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

# Caractère manquant
def is_single_addition_or_deletion(s1, s2):
    """
    Détermine si deux chaînes de caractères diffèrent d'une seule addition ou suppression
    """
    # Conversion explicite en string
    s1, s2 = str(s1), str(s2)
    
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

def has_missing_character(ref, group_refs):
    # Conversion explicite en string
    ref = str(ref)
    group_refs = [str(r) for r in group_refs]
    
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

# Chaine de caractères contenant d'autres références 
def is_contained(ref, group_refs):
    # Conversion explicite en string
    ref = str(ref)
    group_refs = [str(r) for r in group_refs]
    
    for other_ref in group_refs:
        if other_ref != ref:
            pos = other_ref.find(ref)
            if pos != -1:  
                suffix = other_ref[pos + len(ref):] 
                if not (suffix.isalpha() and suffix):  # Si suffixe n'est que des lettres et n'est pas vide, ignorer
                    return True
    return False

def ends_with_alpha(ref):
    return bool(ref) and ref[-1].isalpha()

def update_refs_within_group(group):
    """
    Fonction pour effectuer les affectations au sein des groupes 'Doublons' identiques 
    """
    new_rows = []
    rows_to_remove = set()
    affected_rows = []

    # Assurez-vous que Normalized_Ref existe
    if 'Normalized_Ref' not in group.columns:
        group['Normalized_Ref'] = group['Normalized_Ref_Num'].copy()

    # Cas 1: Traitement des caractères manquants
    for i, row_i in group.iterrows():
        for j, row_j in group.iterrows():
            if i != j and is_single_addition_or_deletion(row_i['Normalized_Ref'], row_j['Normalized_Ref']):
                if len(str(row_i['Normalized_Ref'])) < len(str(row_j['Normalized_Ref'])):
                    new_row = row_i.copy()
                    new_row['Normalized_Ref'] = row_j['Normalized_Ref']
                    new_rows.append(new_row)
                    rows_to_remove.add(i)
                    affected_rows.append((row_i['Normalized_Ref'], row_j['Normalized_Ref']))
    
    # Cas 2: Traitement des références contenues dans d'autres
    for i, row_i in group.iterrows():
        for j, row_j in group.iterrows():
            if i != j and str(row_i['Normalized_Ref']) != str(row_j['Normalized_Ref']) and str(row_i['Normalized_Ref']) in str(row_j['Normalized_Ref']) and not ends_with_alpha(str(row_j['Normalized_Ref'])):
                new_row = row_i.copy()
                new_row['Normalized_Ref'] = row_j['Normalized_Ref']
                new_rows.append(new_row)
                rows_to_remove.add(i)
                affected_rows.append((row_i['Normalized_Ref'], row_j['Normalized_Ref']))

    group = group.drop(index=list(rows_to_remove))

    if new_rows:
        group = pd.concat([group, pd.DataFrame(new_rows)], ignore_index=True)

    return group, affected_rows

def process_dataframe(df):
    """Traite le DataFrame en ajoutant les colonnes normalisées."""
    try:
        # Création d'une barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: Normalisation initiale
        status_text.text("Étape 1/5: Normalisation initiale...")
        df['Normalized_Ref_Num'] = df['Référence'].apply(remove_non_numeric_prefix)
        df['Has_Non_Numeric_Leading'] = df['Référence'].apply(has_non_numeric_leading)
        progress_bar.progress(20)
        
        # Étape 2: Traitement des zéros
        status_text.text("Étape 2/5: Traitement des zéros...")
        df['Normalized_Ref_zeros'] = df['Normalized_Ref_Num'].apply(remove_leading_zeros)
        df['Has_Leading_Zeros'] = df['Normalized_Ref_Num'].apply(has_leading_zeros)
        progress_bar.progress(40)
        
        # Étape 3: Normalisation alphanumérique
        status_text.text("Étape 3/5: Normalisation alphanumérique...")
        df['Normalized_Ref_alpha'] = df['Normalized_Ref_zeros'].apply(remove_non_alphanumeric)
        df['Has_Leading_Alpha'] = df['Normalized_Ref_zeros'].apply(has_special_characters)
        df['Has_Missing_Char'] = False
        df['Is_Contained'] = False
        # Initialisation de la colonne Normalized_Ref
        df['Normalized_Ref'] = df['Normalized_Ref_Num'].copy()
        progress_bar.progress(60)

        # Étape 4: Traitement des remplacements
        status_text.text("Étape 4/5: Traitement des remplacements...")

        # Conteneur pour les messages de remplacement
        replacements_container = st.expander("Détails des remplacements effectués", expanded=False)

        with replacements_container:
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
                            df.loc[(df['Référence'] == alpha) & (df['Doublon'] == doublon_value), 'Normalized_Ref'] = numeric
                            st.write(f"Groupe '{doublon_value}': Remplacement '{alpha}' → '{numeric}'")
                
                # Vérification des caractères manquants
                for index, row in group.iterrows():
                    df.loc[index, 'Has_Missing_Char'] = has_missing_character(row['Référence'], refs)

                # Vérification des références contenues dans d'autres références
                group_refs = group['Normalized_Ref'].tolist()
                for index, row in group.iterrows():
                    df.loc[index, 'Is_Contained'] = is_contained(row['Normalized_Ref'], group_refs)
        
        progress_bar.progress(80)
        
        # Étape 5: Traitement des additions/suppressions
        status_text.text("Étape 5/5: Traitement des additions/suppressions...")
        
        all_new_rows = pd.DataFrame()
        affected_groups = []

        for doublon, group in df.groupby('Doublon', sort=False):
            updated_group, affected_rows = update_refs_within_group(group)
            if not updated_group.empty:
                all_new_rows = pd.concat([all_new_rows, updated_group], ignore_index=True)
            if affected_rows:
                affected_groups.append((doublon, affected_rows))

        if not all_new_rows.empty:
            # Préserver les colonnes qui ne sont pas dans le groupe mis à jour
            for col in df.columns:
                if col not in all_new_rows.columns:
                    all_new_rows[col] = ""  # Initialise avec une chaîne vide
                    # Copie les valeurs de la colonne depuis le df d'origine lorsque possible
                    for idx, row in all_new_rows.iterrows():
                        ref = row['Référence']
                        doublon = row['Doublon']
                        original_row = df[(df['Référence'] == ref) & (df['Doublon'] == doublon)]
                        if not original_row.empty:
                            all_new_rows.loc[idx, col] = original_row[col].values[0]
            df = all_new_rows
        
        with replacements_container:
            for doublon, affected_rows in affected_groups:
                st.write(f"Groupe '{doublon}': Traitements des caractères manquants et références contenues")
                for old_ref, new_ref in affected_rows:
                    st.write(f"Remplacement '{old_ref}' → '{new_ref}'")
                st.write("")

        progress_bar.progress(100)
        status_text.text("Traitement terminé!")

        df['Count'] = df.groupby(['Doublon', 'Normalized_Ref'])['Normalized_Ref'].transform('size')
        df['Doublons Potentiels'] = df['Count'] - 1
        df.drop(columns=['Count'], inplace=True)

        # Sélection des colonnes spécifiques
        columns_to_display = ['Référence', 'Doublon', 'Normalized_Ref', 'Doublons Potentiels']

        # Supposons que votre dataframe s'appelle df
        filtered_df = df[df['Doublons Potentiels'] > 0]

        
        return filtered_df[columns_to_display]
    
    except KeyError as e:
        st.error(f"La colonne '{str(e)}' n'a pas été trouvée dans le fichier.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du traitement : {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def main():
    st.title("🔄 Recherche de potentiels doublons")
    st.markdown("""

    ### Instructions:
    1. Déposez votre fichier Excel (.xlsx)
    2. Lancez le traitement automatique
    3. Téléchargez le résultat
    """)
    
    # Upload du fichier
    uploaded_file = st.file_uploader("📂 Déposez votre fichier Excel", type=["xlsx"])
    
    if uploaded_file:
        try:
            # Lecture du fichier avec message de chargement
            with st.spinner("Lecture du fichier en cours..."):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Affichage des données d'origine
            st.subheader("📊 Aperçu des données d'origine")
            if 'Référence' not in df.columns or 'Doublon' not in df.columns:
                st.error("Le fichier doit contenir les colonnes 'Référence' et 'Doublon'")
                st.dataframe(df.head(), use_container_width=True)
            else:
                st.dataframe(df.head(), use_container_width=True)
                
                # Bouton pour lancer le traitement
                if st.button("🚀 Lancer le traitement"):
                    # Traitement des données
                    processed_df = process_dataframe(df)
                    
                    if processed_df is not None:
                        st.subheader("✨ Résultats du traitement")
                        st.dataframe(processed_df, use_container_width=True)
                        
                        # Export des résultats
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            processed_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Télécharger les résultats",
                            data=output.getvalue(),
                            file_name="resultats_traitement.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Statistiques
                        st.subheader("📈 Statistiques du traitement")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Nombre total de références", len(df))
                        with col2:
                            st.metric("Doublons potentiels", 
                                    int(processed_df['Doublons Potentiels'].sum()))
                
        except Exception as e:
            st.error(f"Une erreur s'est produite: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()