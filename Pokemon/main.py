import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# BACKEND ----------

class PokemonBattlePredictor: # contains winner prediction logic
    
    def __init__(self, data_path: str = 'data/final_pokemon.csv'):
        self.pokemon = self._load_pokemon_data(data_path)
        self.pokemon_names = sorted(self.pokemon['Name'].dropna().unique().tolist())
        

    @st.cache_data
    def _load_pokemon_data(_self, data_path: str) -> pd.DataFrame: # load & pre-process pokemon stats data
        try:
            pokemon = pd.read_csv(data_path)
            pokemon['Name'] = pokemon['Name'].str.strip().str.title()
            logger.info(f"Loaded {len(pokemon)} Pokemon records")
            return pokemon
        except FileNotFoundError:
            st.error(f"Pokemon data file not found: {data_path}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading Pokemon data: {e}")
            st.stop()
    


    def get_pokemon_info(self, name: str) -> Dict[str, Any]: # extract detailed stats for each pokemon
        
        name = name.strip().title()
        pokemon_row = self.pokemon[self.pokemon['Name'] == name]
        
        if pokemon_row.empty:
            return {}
            
        info = pokemon_row.iloc[0].to_dict()
        return info
    


    def get_sprite_url(self, name: str) -> str: # extract sprite image for each pokemon
        try:
            pokemon_info = self.get_pokemon_info(name)
            if 'sprites' not in pokemon_info or pd.isna(pokemon_info['sprites']):
                return ''
            
            # Handle different sprite formats
            sprites = pokemon_info['sprites']
            if isinstance(sprites, str):
                # Try to parse as JSON/dict string
                try:
                    sprite_dict = json.loads(sprites.replace("'", '"'))
                    return sprite_dict.get('normal', sprite_dict.get('front_default', ''))
                except:
                    # If it's already a URL string
                    return sprites if sprites.startswith('http') else ''
            elif isinstance(sprites, dict):
                return sprites.get('normal', sprites.get('front_default', ''))
            
            return ''
        except Exception as e:
            logger.warning(f"Error getting sprite for {name}: {e}")
            return ''
    


    def create_matchup_df(self, name1: str, name2: str) -> pd.DataFrame: # create matchup table for prediction
        
        name1 = name1.strip().title()
        name2 = name2.strip().title()
        
        p1_row = self.pokemon[self.pokemon['Name'] == name1]
        p2_row = self.pokemon[self.pokemon['Name'] == name2]

        if p1_row.empty:
            raise ValueError(f"Pokémon '{name1}' not found in database.")
        if p2_row.empty:
            raise ValueError(f"Pokémon '{name2}' not found in database.")

        p1_id = p1_row.iloc[0]["#"]
        p2_id = p2_row.iloc[0]["#"]

        matchup = pd.DataFrame({
            "First_pokemon": [p1_id],
            "Second_pokemon": [p2_id],
            "Winner": [None]
        })

        merged = matchup.merge(self.pokemon, how="left", left_on="First_pokemon", right_on="#")
        merged = merged.merge(self.pokemon, how="left", left_on="Second_pokemon", right_on="#", suffixes=("_first", "_second"))
        
        # clean up columns
        columns_to_drop = [col for col in [
            '#_first', '#_second', 'sprites_first', 'sprites_second', 
            'Winner', 'First_pokemon', 'Second_pokemon'
        ] if col in merged.columns]
        
        merged = merged.drop(columns=columns_to_drop)
        return merged



    def predict_battle(self, data: pd.DataFrame, model_path: str, encoder_path: str, # predict using model
                      first_pokemon: str, second_pokemon: str) -> Tuple[str, float]:
        """Run the prediction pipeline"""
        
        # validate file paths first
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        try:
            model_df = data.drop(columns=['Name_first', 'Name_second'])

            categorical_features = [
                'Type 1_first', 'Type 2_first', 'Type 1_second', 'Type 2_second',
                'Legendary_first', 'Legendary_second',
                'Generation_first', 'Generation_second'
            ]

            # load & apply encoder
            encoder = joblib.load(encoder_path)
            encoded_data = encoder.transform(model_df[categorical_features])
            encoded_df = pd.DataFrame(
                encoded_data.toarray(),
                columns=encoder.get_feature_names_out(categorical_features),
                index=model_df.index
            )

            model_df_encoded = pd.concat([model_df, encoded_df], axis=1)
            model_df_encoded = model_df_encoded.drop(columns=categorical_features)

            # load model & predict on encoded data
            model = joblib.load(model_path)
            required_columns = list(model.feature_names_in_)

            for col in required_columns: # check if columns exist, zero out in case they don't
                if col not in model_df_encoded.columns:
                    model_df_encoded[col] = 0.0

            model_df_encoded = model_df_encoded[required_columns]

            y_pred_prob = model.predict_proba(model_df_encoded)
            y_pred = model.predict(model_df_encoded)

            winner = first_pokemon if y_pred[0] == 0 else second_pokemon
            confidence = 100 * np.max(y_pred_prob)

            return winner, confidence

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")





# FRONTEND ----------

def setup_page_config():
    st.set_page_config(
        page_title="Pokémon Battle Predictor",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def show_pokemon_card(pokemon_info: Dict[str, Any], sprite_url: str, side: str):
    if not pokemon_info:
        return
        
    color = "1. " if side == "left" else "2. "
    
    # Display sprite
    if sprite_url:
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 10px;">
                <img src="{sprite_url}" width="120" style="border-radius: 10px;"/>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"<div style='text-align: center; padding: 60px; background: #f0f0f0; border-radius: 10px; margin-bottom: 10px;'>No Image</div>", unsafe_allow_html=True)
    
    # Pokemon stats
    st.markdown(f"### {color} {pokemon_info['Name']}")
    
    # Type badges
    type1 = pokemon_info.get('Type 1', 'Unknown')
    type2 = pokemon_info.get('Type 2', None)
    
    type_str = f"**Type:** {type1}"
    if type2 and pd.notna(type2):
        type_str += f" / {type2}"
    st.markdown(type_str)
    
    # Additional info
    st.markdown(f"**Generation:** {pokemon_info.get('Generation', 'Unknown')}")
    st.markdown(f"**Legendary:** {'Yes' if pokemon_info.get('Legendary', False) else 'No'}")
    
    # Individual Pokemon stats
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    with st.expander("📊 Stats", expanded=False):
        for stat in stats:
            if stat in pokemon_info and pd.notna(pokemon_info[stat]):
                st.progress(min(pokemon_info[stat] / 200, 1.0), text=f"{stat}: {pokemon_info[stat]}")



def show_confidence_indicator(confidence: float):
    st.markdown("### Confidence Level")
    
    # Progress bar
    progress_color = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
    st.progress(confidence/100)
    
    # Interpretation
    if confidence >= 80:
        st.success(f"{progress_color} **High Confidence** ({confidence:.1f}%)")
    elif confidence >= 60:
        st.warning(f"{progress_color} **Moderate Confidence** ({confidence:.1f}%)")
    else:
        st.error(f"{progress_color} **Low Confidence** ({confidence:.1f}%)")



# Main Workflow
def main():
    
    setup_page_config()
    
    # Initialize predictor
    try:
        predictor = PokemonBattlePredictor()
    except Exception as e:
        st.error(f"Failed to initialize app: {e}")
        return
    
    # Header
    st.title('⚔️ Pokémon Battle Predictor')
    st.markdown("*Predict the outcome of Pokémon battles using machine learning*")
    st.markdown("---")
    
    # Pokemon selection
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader("Choose Fighter 1")
        first_pokemon = st.selectbox(
            "First Pokémon", 
            predictor.pokemon_names, 
            index=0,
            key="first_pokemon"
        )
        
    with col3:
        st.subheader("Choose Fighter 2")
        second_pokemon = st.selectbox(
            "Second Pokémon", 
            predictor.pokemon_names, 
            index=1,
            key="second_pokemon"
        )
    
    with col2:
        st.markdown(
            """
            <div style="
                display: flex; 
                justify-content: center; 
                padding-top: 50px; 
                font-size: 28px; 
                font-weight: bold;
            ">
                VS
            </div>
            """,
            unsafe_allow_html=True
        )

    
    
    # Prevent same Pokemon selection
    if first_pokemon == second_pokemon:
        st.warning("⚠️ Please select two different Pokémon for battle!")
        return
    
    # Display Pokemon cards
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        p1_info = predictor.get_pokemon_info(first_pokemon)
        p1_sprite = predictor.get_sprite_url(first_pokemon)
        show_pokemon_card(p1_info, p1_sprite, "left")
        
    with col2:
        p2_info = predictor.get_pokemon_info(second_pokemon)
        p2_sprite = predictor.get_sprite_url(second_pokemon)
        show_pokemon_card(p2_info, p2_sprite, "right")
    
    # Model selection
    st.markdown("---")
    st.subheader("Select Prediction Model")
    
    model_choices = {
        'Logistic Regression (Fastest)': 'models/poke_lr.pkl',
        'Random Forest (Balanced)': 'models/poke_rfc.pkl',
        'Gradient Boosting Classifier (Best Performance)': 'models/poke_gbc.pkl'
    }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_model_name = st.selectbox(
            "Choose your model", 
            list(model_choices.keys()), 
            index=2
        )
    
    # Battle button
    st.markdown("---")
    if st.button("⚔️ START BATTLE", type="primary", use_container_width=True):
        with st.spinner("Calculating battle outcome..."):
            try:
                model_path = model_choices[selected_model_name]
                encoder_path = 'models/poke_encoder.pkl'

                df = predictor.create_matchup_df(first_pokemon, second_pokemon)
                winner, confidence = predictor.predict_battle(
                    df, model_path, encoder_path, first_pokemon, second_pokemon
                )

                # Results section
                st.markdown("---")
                st.markdown("## 🏆 Battle Results")
                
                # Winner announcement
                winner_emoji = "1. " if winner == first_pokemon else "2. "
                st.success(f"## **{winner}** WINS")
                
                # Confidence indicator
                show_confidence_indicator(confidence)
                
                # Battle summary
                with st.expander("Battle Summary", expanded=True):
                    st.write(f"**Combatants:** {first_pokemon} vs {second_pokemon}")
                    st.write(f"**Model Used:** {selected_model_name}")
                    st.write(f"**Predicted Winner:** {winner}")
                    st.write(f"**Confidence:** {confidence:.2f}%")

            except FileNotFoundError as e:
                st.error(f"**File Error:** {e}")
                st.info("Please ensure all model files are in the 'models/' directory.")
            except ValueError as e:
                st.error(f"**Data Error:** {e}")
            except RuntimeError as e:
                st.error(f"**Prediction Error:** {e}")
            except Exception as e:
                st.error(f"**Unexpected Error:** {e}")
                logger.exception("Unexpected error during prediction")
    

    # Footer
    st.markdown("---")
    st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
            Predictions based on Pokémon stats and simulated battles.<br>
            Made by <strong>Raafae Zaki</strong> |
            <a href="https://github.com/raafaezaki" target="_blank">GitHub</a>
        </small>
    </div>
    """,
        unsafe_allow_html=True
    )


# Execute main workflow
if __name__ == "__main__":
    main()