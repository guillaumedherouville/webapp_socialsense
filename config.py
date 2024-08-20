# Marketing analyst profiles
profiles = [
    "Name: Alejandro Reyes\
Languages: English, Spanish, Portuguese\
Nationality: Mexican\
Gender: Male\
Age: 38\
Education: MBA in Marketing\
Personality: creative, outgoing, strategic\
Hobbies: film festivals, social media, travel\
Years of working: 15\
Profession: Digital Marketing Specialist\
Specialization: Action and Adventure Films\
Target Audience: Young Adults (18-35)",
    "Name: Yuki Tanaka\
Languages: Japanese, English, Korean\
Nationality: Japanese\
Gender: Female\
Age: 42\
Education: M.S. in Media Studies\
Personality: analytical, detail-oriented, innovative\
Hobbies: anime conventions, blogging, photography\
Years of working: 18\
Profession: Content Marketing Manager\
Specialization: Animated Films and Anime\
Target Audience: Teens and Young Adults",
    "Name: Olivia Bennett\
Languages: English, French\
Nationality: British\
Gender: Female\
Age: 29\
Education: B.A. in Film Studies\
Personality: enthusiastic, adaptable, collaborative\
Hobbies: podcasting, film critique, yoga\
Years of working: 7\
Profession: Social Media Strategist\
Specialization: Indie and Art House Films\
Target Audience: Film Enthusiasts and Critics",
    "Name: Marcus Johnson\
Languages: English\
Nationality: American\
Gender: Male\
Age: 51\
Education: Ph.D. in Communication\
Personality: charismatic, decisive, visionary\
Hobbies: public speaking, mentoring, golf\
Years of working: 25\
Profession: Marketing Director\
Specialization: Blockbusters and Franchise Films\
Target Audience: General Audiences",
    "Name: Priya Patel\
Languages: Hindi, English, Gujarati\
Nationality: Indian\
Gender: Female\
Age: 36\
Education: M.A. in Advertising\
Personality: creative, empathetic, organized\
Hobbies: Bollywood dance, cooking, vlogging\
Years of working: 12\
Profession: Brand Manager\
Specialization: Bollywood and International Films\
Target Audience: South Asian Diaspora",
    "Name: Lars Svensson\
Languages: Swedish, English, Danish\
Nationality: Swedish\
Gender: Male\
Age: 45\
Education: B.S. in Marketing, Film Production Diploma\
Personality: methodical, calm, innovative\
Hobbies: screenwriting, skiing, film restoration\
Years of working: 20\
Profession: Market Research Analyst\
Specialization: Scandinavian and European Cinema\
Target Audience: Art House and International Film Lovers",
    "Name: Zoe Chen\
Languages: Mandarin, English, Cantonese\
Nationality: Singaporean\
Gender: Non-binary\
Age: 33\
Education: M.A. in Digital Marketing\
Personality: tech-savvy, forward-thinking, adaptable\
Hobbies: VR gaming, tech conferences, rock climbing\
Years of working: 10\
Profession: Digital Innovation Specialist\
Specialization: Sci-Fi and Technology-themed Films\
Target Audience: Tech Enthusiasts and Millennials",
    "Name: Isabella Rossi\
Languages: Italian, English, Spanish\
Nationality: Italian\
Gender: Female\
Age: 40\
Education: B.A. in Communications, Culinary Arts Diploma\
Personality: passionate, detail-oriented, persuasive\
Hobbies: wine tasting, film-themed dinner parties, travel blogging\
Years of working: 16\
Profession: Experiential Marketing Expert\
Specialization: Romance and Food-centric Films\
Target Audience: Foodies and Romance Enthusiasts",
    "Name: Ahmed Al-Mansour\
Languages: Arabic, English, French\
Nationality: Egyptian\
Gender: Male\
Age: 37\
Education: M.S. in International Marketing\
Personality: diplomatic, culturally sensitive, strategic\
Hobbies: calligraphy, historical documentaries, chess\
Years of working: 13\
Profession: Cross-Cultural Marketing Specialist\
Specialization: Historical Dramas and Documentaries\
Target Audience: History Buffs and Educational Institutions",
    "Name: Samantha Lee\
Languages: Korean, English\
Nationality: American (Korean descent)\
Gender: Female\
Age: 31\
Education: B.F.A. in Graphic Design, Digital Marketing Certificate\
Personality: creative, trendsetting, analytical\
Hobbies: K-pop fan events, digital art, surfing\
Years of working: 8\
Profession: Influencer Marketing Coordinator\
Specialization: Teen Dramas and Music-themed Films\
Target Audience: Gen Z and K-culture Fans",
]

# Set up comparison DataFrame
barbie = {
    "negative": "24.5%",
    "neutral": "29.6%",
    "positive": "45.8%",
    "sadness": "7.1%",
    "joy": "53.6%",
    "love": "1.6%",
    "anger": "26.6%",
    "fear": "7.8%",
    "surprise": "3.3%",
}
guardians = {
    "negative": "30.8%",
    "neutral": "28.8%",
    "positive": "39.4%",
    "sadness": "14.4%",
    "joy": "47.0%",
    "love": "1.4%",
    "anger": "26.0%",
    "fear": "9.5%",
    "surprise": "1.6%",
}
oppenheimer = {
    "negative": "24.0%",
    "neutral": "30.2%",
    "positive": "45.8%",
    "sadness": "6.8%",
    "joy": "50.3%",
    "love": "1.0%",
    "anger": "31.1%",
    "fear": "8.5%",
    "surprise": "2.3%",
}
flash = {
    "negative": "24.9%",
    "neutral": "30.3%",
    "positive": "44.9%",
    "sadness": "8.9%",
    "joy": "53.1%",
    "love": "1.5%",
    "anger": "26.0%",
    "fear": "8.3%",
    "surprise": "2.1%",
}
MI7 = {
    "negative": "14.8%",
    "neutral": "29.0%",
    "positive": "56.2%",
    "sadness": "7.2%",
    "joy": "57.0%",
    "love": "2.1%",
    "anger": "23.1%",
    "fear": "8.5%",
    "surprise": "2.1%",
}
spider = {
    "negative": "19.6%",
    "neutral": "28.4%",
    "positive": "52.0%",
    "sadness": "6.8%",
    "joy": "49.4%",
    "love": "1.9%",
    "anger": "27.8%",
    "fear": "11.2%",
    "surprise": "2.5%",
}

movies = [barbie, guardians, oppenheimer, flash, MI7, spider]

# Set up comparison DataFrame
Mercury = {
    "negative": "30.2%",
    "neutral": "29.3%",
    "positive": "40.5%",
    "sadness": "10.3%",
    "joy": "46.9%",
    "love": "2.3%",
    "anger": "30.4%",
    "fear": "8.3%",
    "surprise": "1.8%",
}
Sun = {
    "negative": "48.6%",
    "neutral": "32.5%",
    "positive": "18.9%",
    "sadness": "16.1%",
    "joy": "34.9%",
    "love": "1.4%",
    "anger": "37.1%",
    "fear": "9.2%",
    "surprise": "1.4%",
}
Aces = {
    "negative": "37.2%",
    "neutral": "32.5%",
    "positive": "30.3%",
    "sadness": "12.5%",
    "joy": "49.8%",
    "love": "1.3%",
    "anger": "28.5%",
    "fear": "6.8%",
    "surprise": "1.0%",
}
Dream = {
    "negative": "25.2%",
    "neutral": "27.2%",
    "positive": "47.6%",
    "sadness": "8.6%",
    "joy": "57.2%",
    "love": "2.3%",
    "anger": "23.8%",
    "fear": "7.5%",
    "surprise": "0.7%",
}
Sky = {
    "negative": "25.9%",
    "neutral": "26.8%",
    "positive": "47.3%",
    "sadness": "8.5%",
    "joy": "55.4%",
    "love": "1.4%",
    "anger": "26.9%",
    "fear": "5.8%",
    "surprise": "2.0%",
}

games = [Mercury, Sun, Aces, Dream, Sky]
