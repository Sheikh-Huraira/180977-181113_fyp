import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

import Data_Preprocessing
from Data_Preprocessing import PreProcessing
from Streamer import TweetAnalyzer
import joblib
from Tfidf_Test import TFIDF


class PreData():

    def Data_pre(self):
        pre = PreProcessing()
        with open('tweet_text.txt', encoding="utf8") as my_file:
            data = my_file.read()
            d = pre.preprocess(data)
            with open('Processed_data.txt', 'w', encoding="utf8") as pro:
                pro.write(str(d))

    def Calculate_tfidf(self):
        filename = "Processed_data.txt"
        with open(filename, encoding="utf8") as my_file:
            data = my_file.read()
        Tfidf_vect_data = TfidfVectorizer(max_features=40)
        dataa = numpy.ravel(data)
        Tfidf_vect_data.fit(dataa)
        Test_X_Tfidf = Tfidf_vect_data.transform(dataa)
        return Test_X_Tfidf

    def test(self):
        filename = 'finalized_model.sav'
        loaded_model = joblib.load(filename)
        qq = TFIDF()
        daaata = qq.Calculate_tfidf()
        result = loaded_model.predict(daaata)
        print(result)

        if result == 14:
            h1 = "ISTJ: The Inspector"
            p1 = "ISTJ stands for (introversion, sensing, thinking, judgment).People with an ISTJ personality type tend to be reserved, practical and quiet. They enjoy order and organization in all areas of their lives including their home, work, family, and projects. ISTJs value loyalty in themselves and others, and place an emphasis on traditions."
            h2 = "Key ISTJ Characteristics"
            p2 = ("ISTJs are planners; they like to carefully plan things out well in advance. They enjoy an orderly life. They like things to be well-organized and pay a great deal of attention to detail. When things are in disarray, people with this personality type may find themselves unable to rest until they have set everything straight and the work has been completed.",
                  "ISTJs are both responsible and realistic. They take a logical approach to achieving goals and completing projects and are able to work at a steady pace toward accomplishing these tasks. They are able to ignore distractions in order to focus on the task at hand and are often described as dependable and trustworthy.",
                  "ISTJs also place a great deal of emphasis on traditions and laws. They prefer to follow rules and procedures that have previously been established. In some cases, ISTJs can seem rigid and unyielding in their desire to maintain structure.")
            h3 = "Strengths"
            p3 = ("•	Detail-oriented",
                  "•	Realistic",
                  "•	Present-focused",
                  "•	Observant",
                  "•	Logical and practical",
                  "•	Orderly and organized")
            h4 = "Weaknesses:"
            p4 = ("•	Judgmental",
                  "•	Subjective",
                  "•	Tends to blame others",
                  "•	Insensitive")
            h5 = "Dominant: Introverted Sensing"
            p5 = ("•	Introverted sensors are focused on the present moment, taking in an abundance of information about their surroundings.",
                  "•	They also have vivid memories of the past and rely on the memories of these experiences to form expectations for the future.")

            h6 = "Auxiliary: Extraverted Thinking"
            p6 = ("• ISTJs are logical and efficient. They enjoy looking for rational explanations for events",
                  "•	They prefer to focus on the details rather than thinking about abstract information.",
                  "•	Being efficient and productive is important for people with this personality type. They appreciate knowledge that has immediate, practical applications",
                  "•	ISTJs make decisions based on logic and objective data rather than personal feelings.")
            h7 = "Tertiary: Introverted Feeling"
            p7 = ("•	As they make judgments, ISTJs often make personal interpretations based on their internal set of values.",
                  "•	This is often described as an instinct or gut feeling about a situation. ISTJ might make a decision based on logic, only to have this feeling kick in telling them to trust their feelings rather than just the facts")
            h8 = "Personal Relationships"
            p8 = "ISTJs prefer spending time alone or with small groups of close friends. People with this personality type are usually very loyal and devoted to family and friends but may struggle to understand their own emotions and the feelings of others. They can be quite reserved and sometimes fail to pick up on the emotional signals given by other people. However, once they are close to a person and develop an understanding of that person's feelings and needs, they will expend a great deal of effort toward"
            h9 = "Career Paths"

            p9 = ("Because of this need for order, they tend to do better in learning and work environments that have clearly defined schedules, clear-cut assignments and a strong focus on the task at hand. When learning new things, ISTJs do best when the material is something they view as useful with real-world applications. Concrete, factual information appeals to ISTJs, while theoretical and abstract information has little value unless they can see some type of practical use for it. While they may exert tremendous energy into projects they see as valuable, they will avoid wasting time and energy on things that they view as useless or unpractical.",
                  "ISTJs tend to do well in careers that require order, structure, and perseverance. Jobs that involve dealing with concrete facts and figures (accounting, library science, computer programming, etc.) are all good options. Jobs that require accuracy, respect for rules and stability often appeal to those with an ISTJ personality.")
            Bh = "Tips for Interacting With ISTJs:"
            h10 = "Friendships"
            p10 = "ISTJs tend to get along best with friends who are similar to themselves. While they tend to be a bit serious and by the book, they do like to have fun. They might not be willing to jump into new things, but you can be a great friend by helping them pursue hobbies and activities that they enjoy."
            h11 = "Parenting:"
            p11 = ("ISTJ parents tend to be quite focused on tradition and are good at providing security and stability to their children. Children of ISTJ parents often find that their parents will treat them with care and respect and that they also expect the same treatment in return.",
                   "Parents of ISTJ children will find that providing consistency can help their children feel more secure. Sticking to routines, introducing change slowly, and giving them time to adjust to new situations are all ways to help an ISTJ child.")
            h12 = "Relationships"
            p12 = "While ISTJs may experience deep feelings, they often struggle to show that side of themselves in romantic relationships. You can be an understanding partner by not expecting them to bare their soul to you right off the bat. Sometimes it may seem that your partner is not considering your feelings, but you can help them see your side by rationally presenting facts and logical explanations for your side of the argument."
            h13 = "Popular ISTJ Careers"
            p13 = ("•	Accountant",
                   "•	Computer Programmer",
                   "•	Dentist",
                   "•	Doctor",
                   "•	Librarian",
                   "•	Lawyer",
                   "•	Police Officer or Detective",
                   "•	Military Leader")
            h14 = "ISTJs You Might Know"
            p14 = ("•	George Washington, U.S. President",
                   "•	Henry Ford, inventor",
                   "•	Johnny Carson, entertainer",
                   "•	Elizabeth II, Queen of England",
                   "•	Evander Holyfield, boxer")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 12:
            h1 = "ISTP: The Crafter"
            p1 = "ISTP stands for (introverted, sensing, thinking, perceiving).People with ISTP personalities enjoy having time to think alone and are fiercely independent. ISTPs also love action, new experiences, hands-on activities, and the freedom to work at their own pace"
            h2 = "Key ISTP Characteristics"
            p2 = ("People with an ISTP personality are results-oriented. When there is a problem, they want to quickly understand the underlying cause and implement some type of solution. ISTPs are often described as quiet, but with an easy-going attitude towards others.",
                  "ISTPs enjoy new experiences and may often engage in thrill-seeking or even risk-taking behaviors. They often engage in risky or fast-paced hobbies such as motorcycling, hang gliding, bungee jumping, surfing or ice hockey. In some cases, they may seek out adventure by choosing careers in areas such as racing, flying, or firefighting.",
                  "They prefer to make judgments based upon objective criteria rather than personal beliefs or values.",
                  "ISTPs are not well attuned to the emotional states of others, and they can sometimes be seen as a bit insensitive. They also distance themselves from their own emotions, ignoring their feelings until they become overwhelming.",
                  "One common myth about ISTPs is that they are the stoic, silent type. While they do tend to be reserved, this does not mean that they do not experience strong emotions. Instead, they are good at keeping a cool head, maintaining their objectivity, and coping with crisis.")
            h3 = "Strengths"
            p3 = ("Logical",
                  "Learns by experience",
                  "Action-oriented",
                  "Realistic and practical",
                  "Enjoys new things",
                  "Self-confident and easy-going")
            h4 = "Weaknesses:"
            p4 = ("Difficult to get to know",

                  "Insensitive",

                  "Grows bored easily",

                  "Risk-taker",

                  "Does not like commitment")
            h5 = "Dominant: Introverted Thinking"
            p5 = ("•   ISTPs spend a great deal of time thinking and dealing with information in their own heads. This means they do not spend a great deal of time expressing themselves verbally, so they are often known as being quiet",
                  "•	It may seem like the ISTPs approach to decision-making is very haphazard, yet their actions are actually based upon careful observation and thought.",
                  "•   They deal with the world rationally and logically, so they are often focused on things that seem practical and useful.",
                  "•   Because they are so logical, ISTPs are good at looking at situations in an objective way and avoiding subjective or emotional factors when making decisions. People with this personality type can be difficult to get to know, often because they are focused so much on action and results rather than on emotions.")

            h6 = "Auxiliary: Extraverted Sensing"
            p6 = ("• ISTPs prefer to focus on the present and take on things one day at a time. They often avoid making long-term commitments and would rather focus on the here and now rather than think about future plans and possibilities.",
                  "•	ISTPs tend to be very logical and enjoy learning and understanding how things operate. They might take apart a mechanical device just to see how it works.",
                  "•	While they are good at understanding abstract and theoretical information, they are not particularly interested in such things unless they can see some type of practical application.")
            h7 = "Tertiary: Introverted Intuition"
            p7 = ("•	This function often operates largely unconsciously in the ISTP personality. While they are not usually interested in abstract ideas, they may take such concepts and try to turn them into action or practical solutions.",
                  "•	It is this function that is behind the ""gut feelings"" that ISTP sometimes experience when making a decision. By synthesizing information brought in by the dominant and auxiliary functions, this aspect of personality may be responsible for sudden ""aha"" moments of insight")
            h8 = "Personal Relationships"
            p8 = ("ISTPs are introverts and they tend to be quiet and reserved. They thrive on new experiences and dislike strict routines. In relationships, they are highly independent and do not like to feel controlled. Making commitments is difficult for the ISTP, but will put a lot of effort into relationships that hold their interest.",
                  "They do not often share their emotions with other people. While they enjoy hearing what other people think, they frequently keep their own opinions to themselves. For this reasons, people sometimes describe ISTPs as difficult to get to know. They often find friends who enjoy similar hobbies that they do and enjoy spending time with these friends as they pursue these activities.")

            h9 = "Career Paths"

            p9 = "Because ISTPs are introverted, they often do well in jobs that require working alone. ISTPs tend to dislike too much structure and do well in careers where they have a lot of freedom and autonomy. Because they are very logical, they often enjoy work that involves reasoning and hands-on experience. In particular, ISTPs like doing things that have practical, real-world applications."

            Bh = "Tips for Interacting With ISTPs:"
            h10 = "Friendships"
            p10 = "ISTPs tend to be curious and even adventurous, but they also have a strong need to be alone at times. You can be a great friend by asking them to get out and pursue new things, but be ready to respect their need for peace and quiet when they are not feeling up to going out."
            h11 = "Parenting:"
            p11 = "If you are a parent to an ISTP child, you are probably well aware of their independent, adventurous nature. You can encourage their confidence by providing safe and healthy opportunities for them to explore things on their own. Provide rules and guidance, but be careful not to hover. Give your child plenty of hands-on learning, outdoor adventures, and opportunities to experiment with how things work."
            h12 = "Relationships"
            p12 = "Because ISTPs live so strongly in the present moment, long-term commitments can be a real challenge. You can strengthen your relationship with your ISTP partner by being willing to take things day to day and by respecting their fierce need for independence."
            h13 = "Popular ISTP Careers"
            p13 = ("Forensic science",
                   "Engineering",
                   "Mechanics",
                   "Computer programming",
                   "Carpentry",
                   "Law enforcement",
                   "Software engineer",
                   "Video game designer",
                   "Electrician",
                   "Scientist",
                   "Pilot",
                   "Firefighter")
            h14 = "ISTPs You Might Know"
            p14 = ("Clint Eastwood, actor",
                   "Zachary Taylor, U.S. President",
                   "Alan Shepherd, astronaut",
                   "Amelia Earhart, aviator",
                   "Han Solo, Star Wars character")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 8:
            h1 = "INFJ: The Advocate"
            p1 = "INFJ stands for (introverted, intuitive, feeling, and judging). People with INFJ personalities are creative, gentle, and caring. INFJs are usually reserved but highly sensitive to how others feel. They are typically idealistic, with high moral standards and a strong focus on the future. INFJs enjoy thinking about deep topics and contemplating the meaning of life. The INFJ type is said to be one of the rarest with just one to three percent of the population exhibiting this personality type."
            h2 = "Key INFJ Characteristics"
            p2 = (
                "With their strong sense of intuition and emotional understanding, INFJs can be soft-spoken and empathetic. This does not mean that they are push-over's, however. They have deeply held beliefs and an ability to act decisively in order to get what they want.",
                "While they are introverted by nature, people with this personality type are able to form strong, meaningful connections with other people. They enjoy helping others, but they also need time and space to recharge.",
                "While this personality type may be characterized by idealism, this does not mean that INFJs see the world through rose-colored glasses. They understand the world, both the good and the bad, and hope to be able to make it better.",
                "When it comes to making decisions, they place a greater emphasis on personal concerns than objective facts when making decisions. They also like to exert control by planning, organizing and making decisions as early as possible.")
            h3 = "Strengths"
            p3 = ("Sensitive to the needs of others",
                  "Reserved",
                  "Highly creative and artistic",
                  "Focused on the future",
                  "Values close, deep relationships",
                  "Enjoys thinking about the meaning of life",
                  "Idealistic")
            h4 = "Weaknesses:"
            p4 = ("Can be overly sensitive",

                  "Sometimes difficult to get to know",

                  "Can have overly high expectations",

                  "Stubborn",

                  "Dislikes confrontation")
            h5 = "Dominant: Introverted Intuition"
            p5 = (
                "•   This means that they tend to be highly focused on their internal insights.",
                "•	Once they have formed an intuition about something, they tend to stick to it very tightly, often to the point of being single-minded in their focus.",
                "•   Because of this, they are sometimes viewed as being stubborn and unyielding.")

            h6 = "Auxiliary: Extraverted Feeling"
            p6 = (
                "• This characteristic of this type makes INFJs highly aware of what other people are feeling, but it means they are sometimes less aware of their own emotions.",
                "•	INFJs sometimes struggle to say no to other people's requests for this reason. They are so attuned to what other people are feeling that they fear causing disappointment or hurt feelings.")
            h7 = "Tertiary: Introverted Thinking"
            p7 = (
                "•	INFJs make decisions based on ideas and theories that they form based on their own insights.",
                "•	INFJs rely primarily on their introverted intuition and extroverted feeling when making decisions, particularly when they are around other people. When they are alone, however, people with this personality type may rely more on their introverted thinking.",
                "•	In stressful situations, an INFJ might try to rely on emotions when making decisions, especially if it means pleasing other people. Under less stressful conditions, however, an INFJ is more likely to rely more on their intuition.")
            h8 = "Personal Relationships"
            p8 = (
                "INFJs also have a talent for language and are usually quite good at expressing themselves. They have a vivid inner life, but they are often hesitant to share this with others except for perhaps those closest to them. While they are quiet and sensitive, they can also be good leaders. Even when they don't take on overt leadership roles, they often act as quiet influencers behind the scenes.",
                "INFJs are driven by their strong values and seek out meaning in all areas of their lives including relationships and work. People with this type of personality are often described as deep and complex. They may not have a huge circle of acquaintances, but their close friendships tend to be very close and long-lasting.",
                "INFJs are interested in helping others and making the world a better place. They tend to be excellent listeners and are good at interacting with people which whom they are emotionally close and connected. While they care deeply about others, INFJs tend to be very introverted and are only willing to share their (true selves) with a select few. After being in social situations, INFJs need time to themselves to (recharge).")

            h9 = "Career Paths"

            p9 = "INFJs do well in careers where they can express their creativity. Because people with INFJ personality have such deeply held convictions and values, they do particularly well in jobs that support these principles. INFJs often do best in careers that mix their need for creativity with their desire to make meaningful changes in the world."

            Bh = "Tips for Interacting With INFJs:"
            h10 = "Friendships"
            p10 = "Because they are reserved and private, INFJs can be difficult to get to know. They place a high value on close, deep relationships and can be hurt easily, although they often hide these feelings from others. Interacting with an INFJ involves understanding and supporting their need to retreat and recharge. People with this personality type sometimes feel misunderstood. You can be a good friend by taking the time to understand their perspective and appreciating their strengths."
            h11 = "Parenting:"
            p11 = "Because INFJs are so skilled at understanding feelings, they tend to be very close and connected to their children. They have high standards, and can have very high behavioral expectations. They are concerned with raising children that are kind, caring, and compassionate. INFJs encourage their children to pursue their interests and talents in order to fully realize their individual potential."
            h12 = "Relationships"
            p12 = "INFJs have an innate ability to understand other people's feelings and enjoy being in close, intimate relationships. They tend to flourish best in romantic relationships with people who they share their core values. As a partner, it is important to provide the support and emotional intimacy that an INFJ craves. Sincerity, honesty, and authenticity are all traits that the INFJ appreciates in their partner."
            h13 = "Popular INFJ Careers"
            p13 = ("Artist",
                   "Actor",
                   "Entrepreneur",
                   "Religious worker",
                   "Musician",
                   "Librarian",
                   "Counselor",
                   "Psychologist",
                   "Writer",
                   "Teacher",
                   "Photographer")
            h14 = "INFJs You Might Know"
            p14 = ("Oprah Winfey, television personality",
                   "Martin Luther King, Jr., civil rights leader",
                   "Atticus Finch, To Kill a Mockingbird",
                   "Carl Jung, psychoanalyst",
                   "Taylor Swift, musician")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 10:
            h1 = "INTJ: The Architect"
            p1 = "INTJ stands for (introverted, intuitive, thinking, and judging) .People with INTJ personalities are highly analytical, creative and logical.According to psychologist David Keirsey, developer of the Keirsey Temperament Sorter, approximately one to four percent of the population has an INTJ personality type."
            h2 = "Key INTJ Characteristics"
            p2 = (
                "INTJs tend to be introverted and prefer to work alone.",
                "INTJs look at the big picture and like to focus on abstract information rather than concrete details.",
                "INTJs place greater emphasis on logic and objective information rather than subjective emotions.",
                "INTJs like their world to feel controlled and ordered so they prefer to make plans well in advance.")
            h3 = "Strengths"
            p3 = ("Enjoys theoretical and abstract concepts",
                  "High expectations",
                  "Good at listening",
                  "Takes criticism well",
                  "Self-confident and hard-working")
            h4 = "Weaknesses:"
            p4 = ("Can be overly analytical and judgmental",

                  "Very perfectionistic",

                  "Dislikes talking about emotions",

                  "Sometimes seems callous or insensitive")
            h5 = "Dominant: Introverted Intuition"
            p5 = (
                "•   INTJs use introverted intuition to look at patterns, meanings, and possibilities. Rather than simply looking at the concrete facts, they are more interested in what these facts mean.",
                "•	People with this personality type enjoy thinking about the future and exploring possibilities.",
                "•	When remembering events, they may recall impressions more than exact details of what occurred.",
                "•   INTJs are good at (reading between the lines) to figure out what things might really mean.")

            h6 = "Auxiliary: Extraverted Thinking"
            p6 = (
                "• As a secondary function in the INTJ personality, extroverted thinking leads people to seek order, control, and structure in the world around them.",
                "• For this reason, INTJs can be very deliberate and methodical when approaching problems.",
                "•	People with this personality type tend to make decisions based on logic. They organize their thoughts in order to see cause-and-effect relationships.")
            h7 = "Tertiary: Introverted Feeling"
            p7 = (
                "•	INTJs use introverted feeling but because it is a tertiary function, they do so to a lesser degree than they use the dominant and auxiliary functions.",
                "•	Those who develop this aspect of their personalities more fully pay greater attention to values and feelings when making decisions.",
                "•	As a result, they may also feel more drawn to people and activities that are well-aligned with their values.")
            h8 = "Personal Relationships"
            p8 = (
                "People with this personality type are introverted and spend a lot of time in their own mind. INTJs work best by themselves and strongly prefer solitary work to group work.4﻿ While they tend not to be particularly interested in other people's thoughts and feelings, they do care about the emotions of the select group of people to whom they are close. In personal relationships, INTJs are willing to devote time and energy toward making these relationships successful.",
                "Other people often interpret INTJs as cool, aloof and disinterested, which can make forming new friendships challenging. People with this type of personality often see little value in social rituals and small talk, making it even more difficult to get to know them. They tend to be reserved and prefer to interact with a group of close family and friends.")

            h9 = "Career Paths"

            p9 = "When INTJs develop an interest in something, they strive to become as knowledgeable and skilled as they can in that area. They have high expectations, and they hold themselves to the highest possible standards."

            Bh = "Tips for Interacting With INTJs:"
            h10 = "Friendships"
            p10 = "INTJs tend to be solitary and self-sufficient, so establishing friendships can sometimes be difficult. Because people with this personality type tend to think so much of the future, they may avoid getting to know people because they believe that a long-term friendship will not work out. The good news is that while INTJs may not have a lot of friends, they do become very close and committed to those who persist. INTJs tend to prefer friends who are also introverted, rational, and low on emotional drama."
            h11 = "Parenting:"
            p11 = "INTJ parents tend to be thoughtful and attentive, yet they are typically not highly affectionate. They have high expectations for their kids and offer support by helping kids think logically when faced with decisions. Parents with this type of personality encourage their kids to be self-directed critical thinkers who are capable of solving problems on their own."
            h12 = "Relationships"
            p12 = "Because INTJs can be difficult to get to know, romantic relationships can sometimes falter. If your partner has this personality type, it is important to know that loyalty and understanding are important. Don't be afraid to show that you are dedicated to your INTJ partner, but also don't pressure them to spill their emotions. Communication is also critical. Rather than expecting your partner to pick up on your subtle cues, focus on being straight-forward or even blunt about what you expect."
            h13 = "Popular INTJ Careers"
            p13 = ("Scienist",
                   "Mathematician",
                   "Engineer",
                   "Dentist",
                   "Doctor",
                   "Teacher",
                   "Judge",
                   "Lawyer")
            h14 = "INTJs You Might Know"
            p14 = ("Thomas Jefferson, U.S. President",
                   "C.S. Lewis, Author",
                   "Arnold Schwarzenegger, Actor & Politician",
                   "Gandalf, The Lord of the Rings",
                   "Lance Armstrong, Cyclist")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 15:
            h1 = "ISFJ: The Protector"
            p1 = "ISFJ stands for (introverted, sensing, feeling, judging)."
            h2 = "Key ISFJ Characteristics"
            p2 = (
                "ISFJs enjoy structure and strive to maintain this order in all areas of their lives. While people with this personality type are introverted and tend to be quiet, they are keen observers and are focused on other people. Because they are so perceptive, ISFJs are good at remembering details about other people. Those with this personality type are particularly well-tuned into the emotions and feelings of others.",
                "While ISFJs are good at understanding the emotions, they often struggle to express their own feelings. Rather than share their feelings, they may bottle them up, sometimes to the point that negative feelings toward other people can result. When dealing with life struggles such as illness or the death of a loved one, they may keep quiet about what they are experiencing in order to avoid burdening others with their troubles.",
                "People with this personality prefer concrete facts over abstract theories. As a result, they tend to learn best by doing. This also means that they usually value learning for its practical applications. ISFJs tend to become more interested in new things when they can see and appreciate how it might solve a real-world problem.",
                "Because ISFJs tend to be protective of tradition, there is a common myth that they are incapable of change. While people with this personality type may not be quick to change, they are still adaptable. They simply prefer to have time to think about and prepare for big changes.")
            h3 = "Strengths"
            p3 = ("Reliable",
                  "Practical",
                  "Sensitive",
                  "Eye for detail")
            h4 = "Weaknesses:"
            p4 = ("Dislikes abstract concepts",

                  "Avoids confrontation",

                  "Dislikes change",

                  "Neglects own needs")
            h5 = "Dominant: Introverted Sensing"
            p5 = (
                "•  This function leads the introverted sensing types to focus on details and facts. ISFJs prefer concrete information rather than abstract theories. ",
                "•	They are highly attuned to the immediate environment and firmly grounded in reality.",
                "•	Because of this tendency to focus on and protect what is familiar, ISFJs are often seen as highly traditional.",
                "•  When making decisions, ISFJs compare their vivid recall of past experiences in order to predict the outcome of future choices and events. ")

            h6 = "Auxiliary: Extraverted Feeling"
            p6 = (
                "• ISFJs place a great emphasis on personal considerations. Extraverted feelers are focused on developing social harmony and connection. This is accomplished through behaviors that are viewed as socially appropriate or beneficial, such as being polite, kind, considerate, and helpful.",
                "• ISFJs try to fill the wants and needs of other people, sometimes even sacrificing their own desires in order to ensure that other people are happy.")
            h7 = "Tertiary: Introverted Thinking"
            p7 = (
                "•	ISFJs are planners and tend to be very well-organized.",
                "•	This function tends to become stronger as people grow older and involves utilizing logic in order to understand how the world works.",
                "•	As ISFJs take in new information and experiences, they look for connections and commonalities in order to find patterns.",
                "•	Rather than simply trying to understand a small part of something, they want to see how things fit together and how it functions as a whole.")
            h8 = "Personal Relationships"
            p8 = (
                "Because they are quiet, people sometimes misinterpret this as standoffish behavior. As Keirsey notes, this is far from the truth. ISFJs are known for their compassion and caring for others, often working to secure the safety and well-being of other people without asking for thanks or anything in return. While they are introverts, they tend to be warm and quite social.",
                "Because they are hard-working, dependable, and rarely seek accolades for their own accomplishments, ISFJs are sometimes taken for granted by those around them. In some cases, people might even try to take advantage of this reliability.",
                "ISFJs tend to have a small group of very close friends. While they may be quiet and reserved around people they don’t know well, they are more likely to let loose when they are around these close confidants. They place a high value on these close friendships and are always willing to support and care for the people to whom they are close.")

            h9 = "Career Paths"

            p9 = "ISFJs have a number of characteristics that make them well-suited to particular careers. Because they are so attuned to the feelings of others, jobs in mental health or the healthcare industry are a good fit. They are also meticulous and orderly, making them suited to jobs that involve planning, structure, or attention to detail. Their commitment to their work, reliability, and ability to work independently make them attractive to a wide variety of employers."

            Bh = "Tips for Interacting With ISFJs:"
            h10 = "Friendships"
            p10 = "If you are friends with an ISFJ, you are probably already aware that they tend to be warm and selfless. Even though they are quite social for introverts, they are not always good at sharing their own feelings. Asking them how they are doing and being willing to talk can help them to open up. You can help be a good friend by paying attention to their needs. Take the time to see what they might need you to do for them."
            h11 = "Parenting:"
            p11 = (
                "ISFJs are natural caregivers and are very nurturing toward their children. They are good at giving their kids structure and order, but sometimes have a difficult time enforcing discipline.",
                "If you are the parent of an ISFJ child, be aware of your child's need to have time alone. Also be aware that your child may be willing to give up things that are important to them in order to make other people happy. Encourage them to pursue their interests and goals and remind them that meeting their own needs is important as well.")
            h12 = "Relationships"
            p12 = "ISFJs are very faithful to their partners and approach relationships with an intensity of emotion and great devotion. While they have strong feelings, they are not always good at expressing them. Your ISFJ partner may often be focused on taking care of your needs, but you should take care to reciprocate these actions. Showing your partner that you appreciate them can help them to feel more satisfied."
            h13 = "Popular ISFJ Careers"
            p13 = ("Social Worker",
                   "Counselor",
                   "Nurse",
                   "Paralegal",
                   "Bookkeeper",
                   "Child care provider",
                   "Office Manager",
                   "Administratior",
                   "Teacher",
                   "Banker",
                   "Accountant")
            h14 = "ISFJs You Might Know"
            p14 = ("Mother Teresa, nun and humanitarian",
                   "Louisa May Alcott, author",
                   "Kristi Yamaguchi, figure skater",
                   "David Petraeus, U.S. Army General",
                   "Dr. John Watson, Sherlock Holmes series by Arthur Conan Doyle")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 13:
            h1 = "ISFP: The Artist"
            p1 = "ISFP stands for The Artist (Introverted, Sensing, Feeling, Perceiving).People with an ISFP personality are frequently described as quiet, easy-going and peaceful."
            h2 = "Key ISFP Characteristics"
            p2 = (
                "ISFPs like to keep their options open, so they often delay making decisions in order to see if things might change or if new options come up.",
                "According to Myers-Briggs, ISFPs are kind, friendly, sensitive and quiet. Unlike extroverts who gain energy from interacting with other people, introverts must expend energy around others.﻿ After spending time with people, introverts often find that they need a period of time alone. Because of this, they typically prefer to intermingle with a small group of close friends and family members.",
                "While they are quiet and reserved, they are also known for being peaceful, caring, and considerate. ISFPs have an easy-going attitude and tend to accept other people as they are.",
                "ISFPs like to focus on the details. They spend more time thinking about the here and now rather than worrying about the future.",
                "ISFPs tend to be (doers) rather than (dreamers.) They dislike abstract theories unless they can see some type of practical application for them and prefer learning situations that involve gaining hands-on experience.")
            h3 = "Strengths"
            p3 = ("Very aware of their environment",
                  "Practical",
                  "Enjoys hands-on learning",
                  "Loyal to values and beliefs")
            h4 = "Weaknesses:"
            p4 = ("Dislikes abstract, theoretical information",

                  "Reserved and quiet",

                  "Strong need for personal space",

                  "Dislikes arguments and conflict")
            h5 = "Dominant: Introverted Feeling"
            p5 = (
                "•   ISFPs care more about personal concerns rather than objective, logical information.",
                "•	People with this personality type deal with information and experiences based upon how they feel about them.",
                "•   ISFPs have their own value system and create spontaneous judgments based upon how things fit with their own idea.")

            h6 = "Auxiliary: Extraverted Sensing"
            p6 = (
                "• People with ISFP personalities are very in tune with the world around them. They are very much attuned to sensory information and are keenly aware when even small changes take place in their immediate environment. Because of this, they often place a high emphasis on aesthetics and appreciate the fine arts.",
                "•	They are focused on the present moment, taking in new information and then taking action. They have a strong sense of their immediate surroundings, often noticing small details that others might overlook. When remembering events from the past, they are able to recall strong visual imagery and sights, smells, and sounds can evoke powerful memories associated with those senses.")
            h7 = "Tertiary: Introverted Intuition"
            p7 = (
                "•	This function tends to run in the background, feeding off of the extraverted sensing function.",
                "•	As ISFPs take in details about the world, they often develop gut feelings about events and situations. While they generally do not like abstract concepts or ideas, this introverted intuition function may lead them to experience epiphanies about themselves and others.")
            h8 = "Personal Relationships"
            p8 = (
                "ISFPs are introverted. They tend to be reserved and quiet, especially around people they do not know well. They prefer spending time with a close group of family and friends.",
                "ISFPs are very private and keep their true feelings to themselves. In some cases, they may avoid sharing their thoughts, feelings and opinions with other people in their life, even their romantic partners. Because they prefer not to share their innermost feelings and try to avoid conflict, they often defer to the needs or demands of others.",
                "ISFPs have strong values but are not concerned with trying to convince other people to share their point of view. They care deeply about other people, particularly their closest friends and family. They are action-oriented and tend to show their care and concern through action rather than discussing feelings or expressing sentiments.")

            h9 = "Career Paths"

            p9 = "People with ISFP personalities love animals and have a strong appreciation for nature. They may seek out jobs or hobbies that put them in contact with the outdoors and with animals."

            Bh = "Tips for Interacting With ISFPs:"
            h10 = "Friendships"
            p10 = (
                "ISFPs are friendly and get along well with other people, but they typically need to get to know you well before they really open up.",
                "You can be a good friend to an ISFP by being supporting an accepting of who they are.",
                "ISFPs can be light-hearted and fun, but they are also quite intense at times. Recognize that there will be times when your friend wants to share and times when he or she will want to retreat to a more personal space.")
            h11 = "Parenting:"
            p11 = (
                "ISFP children tend to be perfectionists and can be their own harshest critics.",
                "Because they place such high expectations on themselves, they often underestimate or undervalue their own skills and talents.",
                "If you are a parent to ISFP child, you can help your child by encouraging them to be kind to themselves and recognize their value.")
            h12 = "Relationships"
            p12 = (
                "ISFPs are very considerate in relationships, often to the point that they will continually defer to their partner.",
                "Because they are usually not good at expressing their own feelings and needs, it is important that you make an effort to understand your partner.",
                "When making decisions, ensure that your partner's voice is heard and his or her feelings are given equal weight.")
            h13 = "Popular ISFP Careers"
            p13 = ("Artist",
                   "Composer or musician",
                   "Chef",
                   "Designer",
                   "Forest ranger",
                   "Nurse",
                   "Naturalist",
                   "Pediatrician",
                   "Psychologist",
                   "Social worker",
                   "Teacher",
                   "Veterinarian")
            h14 = "ISFPs You Might Know"
            p14 = ("Marilyn Monroe, actress",
                   "Auguste Rodin, sculptor",
                   "David Beckham, soccer player",
                   "Neil Simon, playwright",
                   "Harry Potter, fictional character")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 9:
            h1 = "INFP: The Mediator"
            p1 = "INFP stands for (introversion, intuition, feeling, perception) .The INFP personality type is often described as an (idealist) or (mediator) personality. People with this kind of personality tend to be introverted, idealistic, creative and driven by high values."
            h2 = "Key INFP Characteristics"
            p2 = (
                "INFPs tend to be introverted, quiet, and reserved. Being in social situations tends to drain their energy and they prefer interacting with a select group of close friends. While they like to be alone, this should not necessarily be confused with shyness. Instead, it simply means that INFPs gain energy from spending time alone. On the other hand, they have to expend energy in social situations.",
                "INFPs typically rely on intuition and are more focused on the big picture rather than the nitty-gritty details. They can be quite meticulous about things they really care about or projects they are working on, but tend to ignore mundane or boring details.",
                "INFPs place an emphasis on personal feelings and their decisions are more influenced by these concerns rather than by objective information.",
                "When it comes to making decisions, INFPs like to keep their options open. They often delay making important decisions just in case something about the situation changes. When decisions are made, they are usually based on personal values rather than logic.")
            h3 = "Strengths"
            p3 = ("Loyal and devoted",
                  "Sensitive to feelings",
                  "Caring and interested in others",
                  "Works well alone",
                  "Values close relationships",
                  "Good at seeing the big picture")
            h4 = "Weaknesses:"
            p4 = ("Can be overly idealistic",

                  "Tends to take everything personally",

                  "Difficult to get to know",

                  "Sometimes loses sight of the little things",

                  "Overlooks details")
            h5 = "Dominant: Introverted Feeling"
            p5 = "INFPs experience a great depth of feelings, but as introverts they largely process these emotions internally. They possess an incredible sense of wonder about the world and feel great compassion and empathy for others. While these emotions are strong, they tend not to express them outwardly, which is why they can sometimes be mistaken as aloof or unwelcoming."

            h6 = "Auxiliary: Extraverted Intuition"
            p6 = "INFPs explore situations using imagination and 'what if' scenarios, often thinking through a variety of possibilities before settling on a course of action. Their inner lives are a dominant force in personality and they engage with the outside world by using their intuition. They focus on the big picture and things will shape the course of the future. This ability helps make INFPs transformative leaders who are excited about making positive changes in the world."
            h7 = "Tertiary: Introverted Sensing"
            p7 = "When taking in information, INFPs create vivid memories of the events. They will often replay these events in their minds to analyze experiences in less stressful settings. Such memories are usually associated with strong emotions, so recalling a memory can often seem like reliving the experience itself."
            h8 = "Personal Relationships"
            p8 = (
                "INFP are idealists so they tend to have high expectations - including in relationships. They might hold an idealized image in their minds of their perfect partner, which can be a difficult role for any individual to fill.",
                "People with this personality type care deeply about other people, yet as introverts they can be difficult to know. They do tend to become very close and deeply committed to the few that they forge close relationships with.",
                "They also dislike conflict and try to avoid it. When conflicts or arguments do arise, they usually focus more on how the conflict makes them feel rather than the actual details of the argument. During arguments, they might seem overly emotional or even irrational. However, they can also be good mediators by helping the people involved in a conflict identify and express their feelings.",
                "Because they are so reserved and private, it can be difficult for other people to get to know INFPs. They tend to be quite devoted to their circle of close friends and family and place a high importance on the feelings and emotions of their loved ones.﻿ Much of their energy is focused inwardly and characterized by intense feelings and strong values. They tend to be very loyal to the people they love and to beliefs and causes that are important to them.")

            h9 = "Career Paths"

            p9 = "INFPs typically do well in careers where they can express their creativity and vision. While they work well with others, they generally prefer to work alone. "

            Bh = "Tips for Interacting With INFPs:"
            h10 = "Friendships"
            p10 = "INFPs typically only have a few close friendships, but these relationships tend to be long-lasting. While people with this type of personality are adept at understanding others emotions, they often struggle to share their own feelings with others. Social contact can be difficult, although INFPs crave emotional intimacy and deep relationships. Getting to known an INFP can take time and work, but the rewards can be great for those who have the patience and understanding."
            h11 = "Parenting:"
            p11 = "INFP parents are usually supportive, caring, and warm. They are good at establishing guidelines and helping children develop strong values.﻿ Their goal as parents is to help their children grow as individuals and fully appreciate the wonders of the world. They may struggle to share their own emotions with their children and are often focused on creating harmony in the home."
            h12 = "Relationships"
            p12 = "As with friendships, INFPs may struggle to become close to potential romantic partners. Once they do form a relationship, they approach it with a strong sense of loyalty. They can sometimes hold overly romanticized views of relationships and may have excessively high expectations that their partners struggle to live up to. They also tend to take comments personally while at the same time struggling to avoid conflicts. If your partner is an INFP, understand that they may struggle at times to open up, be overly sensitive to perceived criticisms, and often place your own happiness over that of their own."
            h13 = "Popular INFP Careers"
            p13 = ("Artist",
                   "Counselor",
                   "Graphic Designer",
                   "Librarian",
                   "Psychologist",
                   "Physical Therapist",
                   "Social Worker",
                   "Writer")
            h14 = "INFPs You Might Know"
            p14 = ("Audrey Hepburn, actress",
                   "JRR Tolkien, author",
                   "Princess Diana, British royal",
                   "William Shakespeare, playwright",
                   "Fred Rogers, television personality")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14

        elif result == 11:
            h1 = "INTP: The Thinker"
            p1 = "INTP stands for (introverted, intuitive, thinking, perceiving). People who score as INTP are often described as quiet and analytical. They enjoy spending time alone, thinking about how things work and coming up with solutions to problems. INTPs have a rich inner world and would rather focus their attention on their internal thoughts rather than the external world. They typically do not have a wide social circle, but they do tend to be close to a select group of people. "
            h2 = "Key INTP Characteristics"
            p2 = (
                "INTPs are quiet, reserved, and thoughtful. As introverts, they prefer to socialize with a small group of close friends with whom they share common interests and connections.",
                "They enjoy thinking about theoretical concepts and tend to value intellect over emotion. INTPs are logical and base decisions on objective information rather than subjective feelings.",
                "When analyzing data and making decisions, they are highly logical and objective.",
                "Tends to be flexible and good at thinking outside of the box.",
                "People with this personality type think about the big picture rather than focusing on every tiny detail.",
                "INTPs like to keep their options open and feel limited by structure and planning.")
            h3 = "Strengths"
            p3 = ("Logical and objective",
                  "Abstract thinker",
                  "Independent",
                  "Loyal and affectionate with loved ones")
            h4 = "Weaknesses:"
            p4 = ("Difficult to get to know",

                  "Can be insensitive",

                  "Prone to self-doubt",

                  "Struggles to follow rules",

                  "Has trouble expressing feelings")
            h5 = "Dominant: Introverted Thinking"
            p5 = "This function focuses on how people take in information about the world. INTPs express this by trying to understand how things work. They often like to break down larger things or ideas in order to look at the individual components in order to see how things fit and function together. INTPs tend to be highly logical and efficient thinkers. They like to have a complete understanding of something before they are willing to share an opinion or take action."

            h6 = "Auxiliary: Extraverted Intuition"
            p6 = "INTPs express this cognitive function by exploring what-ifs and possibilities. They utilize insight, imagination, and past experiences to form ideas. They often go over what they know, seeking patterns until they are able to achieve a flash of inspiration or insight into the problem. They tend to spend a great deal of time thinking about the future and imagining all the possibilities."
            h7 = "Tertiary: Introverted Sensing"
            p7 = "INTPs tend to be very detail-oriented, carefully categorizing all of the many facts and experiences that they take in. As they collect new information, they compare and contrast it with what they already know in order to make predictions about what they believe will happen next."
            h8 = "Personal Relationships"
            p8 = (
                "As introverts, INTPs prefer spending time alone for the most part. Unlike extraverts, who gain energy from interacting with a wide group of people, introverts must expend energy in social situations. After being around a lot of people, INTPs might feel like they need to spend some time alone to recharge and find balance. While they may be shy around people they do not know well, INTPs tend to be warm and friendly with their close group of family and friends.",
                "Because INTPs enjoy solitude and deep thinking, they sometimes strike others as aloof and detached. At times, people with this personality type can get lost in their own thoughts and lose track of the outside world. They love ideas and place a high value on intelligence and knowledge.",
                "In social situations, INTPs tend to be quite easy-going and tolerant. However, they can become unyielding when their beliefs or convictions are challenged. Their high emphasis on logic can make it difficult to not correct others in situations where other people present arguments that are not rational or logical. Because they rely on their own minds rather than others, they can also be very difficult to persuade.")

            h9 = "Career Paths"

            p9 = "Because they enjoy theoretical and abstract concepts, INTPs often do particularly well in science-related careers. They are logical and have strong reasoning skills, but are also excellent at thinking creatively."

            Bh = "Tips for Interacting With INTPs:"
            h10 = "Friendships"
            p10 = "Shared interests are one of the best paths to forming a friendship with an INTP. They tend to value intellect over all else and can be very slow to form friendships. While this often leads to fewer friendships, the ones that an INTP does gain tend to be very close. Remember that you INTP friends may not be the best at dealing with excess emotions, but they love to bond over deep conversations and shared passions."
            h11 = "Parenting:"
            p11 = "If your child is an INTP, it is important to remember that your child may respond better to reason and logic rather than appeals to emotion. Encourage your child to develop his or her intellectual interest, but also look for situations that may help your child foster friendships. This can be an area where your child struggles, but putting them in contact with other kids who share the same interests can be helpful."
            h12 = "Relationships"
            p12 = "INTPs tend to live inside their minds, so they can be quite difficult to get to know. Even in romance, they often hold back until they feel that the other person has proven themselves worthy of hearing these innermost thoughts and feelings. One thing to remember is that while INTPs do enjoy romance in the context of a deeply committed relationship, they do not play games. Be honest and forthright. Because INTPs are not good at understanding the emotional needs of others, you may need to be very direct about what you need and expect in that regards. INTPs also struggle to share their own feelings, so you may need to pay attention to subtle signals that your partner is sending."
            h13 = "Popular INTP Careers"
            p13 = ("Chemist",
                   "Physicist",
                   "Computer Programmer",
                   "Forensic Scientist",
                   "Engineer",
                   "Mathematician",
                   "Pharmacist",
                   "Software Developer",
                   "Geologist")
            h14 = "INTPs You Might Know"
            p14 = ("Albert Einstein, scientist",
                   "Dwight D. Eisenhower, U.S. President",
                   "Carl Jung, psychoanalyst",
                   "Tiger Woods, golfer",
                   "Sheldon Cooper, The Big Bang Theory")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 7:
            h1 = "ESTP: The Persuader"
            p1 = "ESTP stands for(Extraverted, Sensing, Thinking, Perceiving).People with this personality type are frequently described as outgoing, action-oriented and dramatic. ESTPs are outgoing and enjoy spending time with a wide circle of friends and acquaintances. They are interested in the here-and-now and are more likely to focus on details than taking a broader view of things."
            h2 = "Key ESTP Characteristics"
            p2 = (
                "When confronted by problems, people with this personality type quickly look at the facts and devise an immediate solution. They tend to improvise rather than spend a great deal of time planning.",
                "ESTPs don't have a lot of use for abstract theories or concepts. They are more practical, preferring straightforward information that they can think about rationally and act upon immediately.",
                "They are very observant, often picking up on details that other people never notice. Other people sometimes describe them as fast-talkers who are highly persuasive. In social settings, they often seem like they are a few steps ahead of the conversation.",
                "ESTPs are not planners. They react in the moment and can often be quite impulsive or even risk-taking. This leap before they look attitude can be problematic at times and it may lead them to saying or doing things that they wish they could take back.",
                "One common myth about ESTPs is that they are reckless. In some instances, people with this personality type can veer into reckless behavior. In most cases, however, ESTPs act quickly based on their impressions and logic.")
            h3 = "Strengths"
            p3 = ("Gregarious, funny, and energetic",
                  "Influential and persuasive",
                  "Action-oriented",
                  "Adaptable and resourceful",
                  "Observant")
            h4 = "Weaknesses:"
            p4 = ("Impulsive",

                  "Competitive",

                  "Dramatic at times",

                  "Easily bored",

                  "Insensitive")
            h5 = "Dominant: Extraverted Sensing"
            p5 = (
                "• Because they are so focused on the present world, ESTPs tend to be realists. They are interested in the sights, sounds, and experiences that are going on immediately around them, and they have little use for daydreams or flights of fancy.  ",
                "•	As sensors, people with this personality type want to touch, feel, hear, taste and see anything and everything that might possibly draw their interest. When learning about something new, it's not just enough to read about it in a textbook or listen to a lecture – they want to experience it for themselves.",
                "•  ESTPs also have lots of energy, so they can become bored in situations that are tedious or in learning situations that involve a great deal of theoretical information. ESTPs are the quintessential doers – they get straight to work and are willing to take risks in order to get the job done. ")

            h6 = "Auxiliary: Introverted Thinking"
            p6 = (
                "• When making judgments about the world, ESTPs focus inwardly where they process information in a logical and rational way. Because this side of personality is introverted, it is something that people may not immediately notice.",
                "• This inner sense of control gives ESTPs a great deal of self-discipline. They are skilled at working independently and can be very goal-directed when they want to achieve an objective.",
                "• They have excellent observational skills, noticing things that others may overlook. As they take in information, they then apply their sense of logic to look for practical and immediately applicable solutions.")
            h7 = "Tertiary: Extraverted Feeling"
            p7 = (
                "•	This function focuses on creating social harmony and relationships with others. While emotions are not an ESTPs strongest suit, they do have a great need for social engagement. They enjoy being at the center of attention and are good at establishing a friendly rapport with other people.",
                "•	While they are social, ESTPs are sometimes less comfortable sharing their opinions and judgments with others. Rather than rock the boat, they are more focused on pleasing others and maintaining harmony. They may overlook their own needs at times to ensure that other people are happy.")
            h8 = "Personal Relationships"
            p8 = "As extroverts, ESTPs gain energy from being around other people. In social settings, people with this personality type are seen as fun, friendly and charming. According to Keirsey, people with this personality type are particularly skilled at influencing people. ESTPs are not only great at interacting with other people, they have a natural ability to perceive and interpret nonverbal communication. Thanks to these abilities, ESTPs tend to do very well in careers that involve sales and marketing."

            h9 = "Career Paths"

            p9 = "People with an ESTP personality type feel energized when they interact with a wide variety of people, so they do best in jobs that involve working with others. They strongly dislike routine and monotony, so fast-paced jobs are ideal."

            Bh = "Tips for Interacting With ESTPs:"
            h10 = "Friendships"
            p10 = "ESTPs have an inexhaustible thirst for adventure. You can be a good friend by always being ready to head out for a new experience, or even by coming up with plans that offer excitement, novelty, and challenge.",
            h11 = "Parenting:"
            p11 = "ESTP children can be adventurous and independent, which is why parents need to set boundaries and ensure that fair, consistent discipline is used. Kids with this type of personality needs lots of hand-on activities to keep them busy, but they may struggle in classroom settings where they quickly grow weary of routines.",
            h12 = "Relationships"
            p12 = "ESTPs are exciting and fun-loving, but they can grow bored with routines quickly. They do not enjoy long, philosophical discussions but like to keep the conversation flowing as they talk about shared interests and passions. Be aware that your partner prefers to take things day by day, may struggle with making long-term commitments, and has a hard time making plans for the future.",
            h13 = "Popular ESTP Careers"
            p13 = ("Sales Agent",
                   "Marketer",
                   "Entrepreneur",
                   "Police Officers",
                   "Detectives",
                   "Computer Support Technician",
                   "Paramedic")
            h14 = "ESTPs You Might Know"
            p14 = ("Donald Trump, businessman and U.S. President",
                   "Madonna, singer",
                   "Ernest Hemingway, novelist",
                   "Thomas Edison, inventor",
                   "Captain James T. Kirk, fictional character, Star Trek")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 5:
            h1 = "ESFP: The Performer"
            p1 = "ESFP stands for (extraverted, sensing, feeling, perceiving). People with ESFP personality types are often described as spontaneous, resourceful, and outgoing. They love being the center of attention and are often described as entertainers or “class clowns.” ESFP is the opposite of the INTJ personality type."
            h2 = "Key ESFP Characteristics"
            p2 = (
                "ESFPs tend to be very practical and resourceful. They prefer to learn through hands-on experience and tend to dislike book learning and theoretical discussions. Because of this, students with ESFP personality types sometimes struggle in traditional classroom settings. However, they excel in situations where they are allowed to interact with others or learn through direct experience.",
                "ESFPs live very much in the here-and-now and sometimes fail to think about how current actions will lead to long-term consequences. They will often rush into a new situation and figure things out as they happen. They also tend to dislike routine, enjoy new experiences, and are always looking for a new adventure.",
                "In addition to having a strong awareness of their surroundings, they are also very understanding and perceptive when it comes to other people. They are able to sense what others are feeling and know how to respond. People tend to find them warm, sympathetic, and easygoing.",
                "One common myth about ESFPs is that they are attention-seekers. While they are fun-loving and do not shun the spotlight, they are more interested in simply living in the present and doing what feels right at that moment.")
            h3 = "Strengths"
            p3 = ("Optimistic and gregarious",
                  "Enjoys people and socializing",
                  "Focused on the present, spontaneous",
                  "Practical")
            h4 = "Weaknesses:"
            p4 = ("Dislikes abstract theories",

                  "Becomes bored easily",

                  "Does not plan ahead",

                  "Impulsive")
            h5 = "Dominant: Extraverted Sensing"
            p5 = (
                "• ESFPs prefer to focus on the here-and-now rather than thinking about the distant future. They also prefer learning about concrete facts rather than theoretical ideas.  ",
                "•	ESFPs don’t spend a lot of time planning and organizing. Instead, they like to keep their options open.",
                "•  When solving problems, they trust their instincts and put trust in their own abilities to come up with a solution. While they are reasonable and pragmatic, they dislike structure, order, and planning. Instead, they act spontaneously and do not spend a great deal of time coming up with a plan or schedule. ")

            h6 = "Auxiliary: Introverted Feeling"
            p6 = (
                "• ESFPs place a greater emphasis personal feelings rather than logic and facts when making decisions.",
                "• People with this personality type have an internal system of values on which they base their decisions. They are very much aware of their own emotions and are empathetic towards others. They excel at putting themselves in another person's shoes, so to speak.")
            h7 = "Tertiary: Extraverted Thinking"
            p7 = (
                "•	This function is focused on enforcing order on the outside world. It is centered on productivity, logic, and results.",
                "•	Because this tends to be a weaker aspect of personality, ESFPs may not always feel secure sharing their judgments, especially if they feel it will disrupt the harmony of the group.")
            h8 = "Personal Relationships"
            p8 = (
                "As extroverts, ESFPs enjoy spending time with other people and have excellent interpersonal skills.2﻿ They are good at understanding how other people are feeling and are able to respond to other people's emotions in productive ways. For this reason, ESFPs can make good leaders and have a knack for mobilizing, motivating and persuading group members.",
                "ESFPs are often described as warm, kind and thoughtful, making them popular and well-liked by others. ESFPs enjoy meeting new people, but they also have a thirst for new experiences. They are generally focused on the present and will often be the first person to try the newest ride at an amusement park or try out a new adventure sport.")

            h9 = "Career Paths"

            p9 = "With their strong dislike for routine, ESFPs do best in careers that involve a lot of variety. Jobs that involve a great deal of socializing are also a great fit, allowing individuals with this personality type to put their considerable people skills to good use. Careers that involve a great deal of structure and solitary work can be difficult for ESFPs, and they often become bored in such situations."

            Bh = "Tips for Interacting With ESFPs:"
            h10 = "Friendships"
            p10 = "ESFPs grow weary with the same old routines and are always ready for a new adventure. In order to keep up with this personality type, you need to always be ready for new experiences - from exploring new places to meeting new people. Keeping things interesting is important, but ESFPs love to have a reliable co-conspirator who is as ready for fun as they are.",
            h11 = "Parenting:"
            p11 = "ESFP children are enthusiastic and energetic, which can be both fun and exhausting for parents. You can help by providing plenty of outlets for this boundless energy. Sports, hobbies, and outdoor adventures are all good sources of fun for ESFP kids. While these kids are people-loving extroverts, they may need time alone to process their feelings when they are upset. Be sure to give them some time before drawing them out to discuss their emotions.",
            h12 = "Relationships"
            p12 = "ESFPs tend to be honest and forthright in relationships. They don't play games and are warm and enthusiastic in romantic relationships. One thing to remember is that ESFPs dislike conflict and tend to take any critical comments quite personally. While it is important to be straightforward in your relationship with an ESFP, try to avoid being overly harsh or confrontational.",
            h13 = "Popular ESFP Careers"
            p13 = ("Artist",
                   "Actor",
                   "Counselor",
                   "Social Worker",
                   "Athletic coach",
                   "Child care provider",
                   "Musician",
                   "Psychologist",
                   "Human Resources Specialist",
                   "Fashion Designer")
            h14 = "ESFPs  You Might Know"
            p14 = ("Bill Clinton, U.S. President",
                   "Pablo Picasso, artist",
                   "Mark Cuban, entrepreneur",
                   "Will Smith, actor",
                   "Fred and George Weasley, fictional characters from Harry Potter")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 1:
            h1 = "ENFP: The Champion"
            p1 = "ENFP stands for(Extraverted, Intuitive, Feeling, Perceiving).People with this type of personality are often described as enthusiastic, charismatic, and creative. People with this personality type are very charming, energetic, and independent. They are creative and do best in situations where they have the freedom to be creative and innovative. An estimated 5 to 7 percent of people are ENFPs."
            h2 = "Key ENFP Characteristics"
            p2 = (
                "ENFPs have excellent people skills. In addition to having an abundance of enthusiasm, they also genuinely care about others. ENFPs are good at understanding what other people are feeling. Given their zeal, charisma, and creativity, they can also make great leaders.",
                "People with this personality type strongly dislike routine and prefer to focus on the future. While they are great at generating new ideas, they sometimes put off important tasks until the last minute. Dreaming up ideas but not seeing them through to completion is a common problem.",
                "ENFPs can also become easily distracted, particularly when they are working on something that seems boring or uninspiring.",
                "ENFPs are flexible and like to keep their options open. They can be spontaneous and are highly adaptable to change. They also dislike routine and may have problems with disorganization and procrastination.")
            h3 = "Strengths"
            p3 = ("Warm and enthusiastic",
                  "Empathetic and caring",
                  "Strong people skills",
                  "Strong communication skills",
                  "Fun and spontaneous",
                  "Highly creative")
            h4 = "Weaknesses:"
            p4 = ("Needs approval from others",

                  "Disorganized",

                  "Tends to get stressed out easily",

                  "Can be overly emotional",

                  "Overthinks",

                  "Struggles to follow rules")
            h5 = "Dominant: Extraverted Intuition"
            p5 = "ENFPs generally focus on the world of possibilities. They are good at abstract thinking and prefer not to concentrate on the tiny details. They are inventive and focused on the future. ENTPs are good at seeing things as they might be rather than focusing simply what they are. They have a natural tendency to focus on relationships and are skilled at finding patterns and connections between people, situations, and ideas."

            h6 = "Auxiliary: Introverted Feeling"
            p6 = "When making decisions, ENFPs place a greater value on feelings and values rather than on logic and objective criteria. They tend to follow their heart, empathize with others, and let their emotions guide their decisions. ENTPs have a strong desire to be true to themselves and their values. In an ideal world, their the world would be in congruence with their values."
            h7 = "Tertiary: Extraverted Thinking"
            p7 = "This cognitive function is centered on organizing information and ideas in a logical way. When looking at information, the ENTP may use this function to sort through disparate data in order to efficiently spot connections. For example, an ENTP might think out loud as they are working through a problem, laying out all the information in order to create an easily followed train of thought."
            h8 = "Personal Relationships"
            p8 = (
                "ENFPs are extroverts, which means that they love spending time with other people.Socializing actually gives them more energy, helping them to feel renewed, refreshed, and excited about life. While other types of extraverts tend to dislike solitude, ENFPs do have a need for some alone time in order to think and reflect.",
                "ENTPs tend to be warm and passionate in relationships. As extraverts, they are naturally upbeat and gregarious. In relationships, they are always seeking growth and ways to make their partnerships stronger. They tend to be attentive and spontaneous. Their willingness to take risks can sometimes be stressful for those who love them.")

            h9 = "Career Paths"

            p9 = "When choosing a career path, it is a good idea for people to understand the potential strengths and weaknesses of their personality type. People with the ENFP personality type do best in jobs that offer a lot of flexibility."

            Bh = "Tips for Interacting With INTJs:"
            h10 = "Friendships"
            p10 = "ENFPs make fun and exciting friends. They enjoy doing new things and usually have a wide circle of friends and acquaintances. They are perceptive of other people feelings and are good at understanding other people quite quickly.You can help your ENFP friends by providing the emotional support to help them achieve their goals."
            h11 = "Parenting:"
            p11 = (
                "Because ENFPs dislike routine, their children may sometimes perceive them as inconsistent. However, they typically have strong, loving relationships with their kids and are good at imparting their sense of values. Parents of ENFP children will find that their child has a strong sense of imagination and a great deal of enthusiasm for life. Your child's energy may seem overwhelming at times, but you should look for ways to help your child explore their creativity.",
                "One struggle they may face is with providing structure and limits. While they recognize the needs for such things, they are not always good at setting or enforcing such limitations. Parents of ENTPs should encourage their kids to be creative, but provide rules and guidelines.")
            h12 = "Relationships"
            p12 = "ENFPs tend to be passionate and enthusiastic in romantic relationships. Long-term relationships can sometimes hit a snag because people with this personality type are always thinking about what is possible rather than simply focusing on things as they are. In order to keep the romance alive, it is important to look for new ways to bring excitement into the relationship."
            h13 = "Popular ENFP Careers"
            p13 = ("Psychologist",
                   "Journalist",
                   "Actor",
                   "TV Anchor/Reporter",
                   "Nutritionist",
                   "Nurse",
                   "Social Worker",
                   "Politician",
                   "Counselor")
            h14 = "ENFPs You Might Know"
            p14 = ("Andy Kaufmann, comedian",
                   "Dr. Seuss, children's author",
                   "Salvador Dali, artist",
                   "Ellen Degeneres, comedian and talk show host",
                   "Ron Weasley, Harry Potter")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 3:
            h1 = "ENTP: The Debater"
            p1 = "ENTP stands for(Extraverted, Intuitive, Thinking, Perceiving). People with this personality type are often described as innovative, clever, and expressive. ENTPs are also known for being idea-oriented, which is why this personality type has been described as the innovator,the visionary, and the debater."
            h2 = "Key ENTP Characteristics"
            p2 = (
                "ENTPs enjoy interacting with a wide variety of people. They are great conversationalists and love to engage other people in debates.",
                "They are more focused on the future rather than on immediate details. They may start projects and never finish them because they are so focused on the big picture rather than the present needs.",
                "ENTPs enjoy being around other people, particularly if they are able to engage in a conversation or debate about something in which they are interested. They are usually fairly laid-back and easy to get along with. However, they can sometimes get so wrapped up in their ideas or plans that they lose sight of their close relationships.",
                "They tend to reserve judgment. Instead of making a decision or committing to a course of action, they would prefer to wait and see what happens.",
                "ENTPs are immensely curious and focused on understanding the world around them. They are constantly absorbing new information and ideas and quickly arriving at conclusions. They are able to understand new things quite quickly.",
                "One common myth about ENTPs is that they love to argue simply for the sake of arguing. While people with this personality type are often willing to play the devil's advocate at times, they enjoy debates as a way of exploring a topic, learning what other people believe, and helping others see the other side of the story.")
            h3 = "Strengths"
            p3 = ("Innovative",
                  "Creative",
                  "Great conversationalist",
                  "Enjoys debating",
                  "Values knowledge")
            h4 = "Weaknesses:"
            p4 = ("Can be argumentative",

                  "Dislikes routines and schedules",

                  "Does not like to be controlled",

                  "Unfocused",

                  "Insensitive")
            h5 = "Dominant: Extraverted Intuition"
            p5 = (
                "•  ENTPs tend to take in information quickly and are very open-minded. ",
                "•	Once they have gathered this information, they spend time making connections between various complex and interwoven relationships.",
                "•	They are good at spotting connections that others might overlook and tend to be focused on possibilities.",
                "•  They have entrepreneurial minds and are always coming up with new and exciting ideas. ")

            h6 = "Auxiliary: Introverted Thinking"
            p6 = (
                "• This cognitive function is expressed in the ENTPs thinking process. People with this type of personality are more focused on taking in information about the world around them. When they do use this information to reach conclusions, they tend to be very logical.",
                "• ENTPs are logical and objective. When making decisions, they place a greater weight on rational evidence instead of subjective, emotional information.",
                "• This function works to help the ENTP understand all the information that comes in through the extraverted intuition function. This involves imposing logic and order to help make sense of many disparate ideas and pieces of information. ENTPs don't want to just understand that something works - they want to understand the why and how behind how things function.")
            h7 = "Tertiary: Extraverted Feeling"
            p7 = (
                "•	As a tertiary function, this aspect of personality may not be as well-developed or prominent.",
                "•	When developed, ENTPs can be social charmers who are able to get along well with others.",
                "•	When this aspect of personality is weaker, the ENTP may be insensitve to others and can even be seen as aloof or unkind.")
            h8 = "Personal Relationships"
            p8 = (
                "Since they are identified as extraverts, it may come as no surprise that ENTPs have very good people skills. They are skilled communicators and enjoy interacting with a wide circle of family, friends, and acquaintances. In conversations, other people often find them quick-witted.",
                "ENTPs will often engage in debates simply because they enjoy having a good battle of the wits. Sometimes, their love of debates lead ENTPs to take on the role of the devil's advocate, which can sometimes lead to conflicts with others who feel like they are being intentionally combative and antagonistic.")

            h9 = "Career Paths"

            p9 = "Routines and boredom are not good for the ENTP personality. They are non-conformists and do best in jobs when they can find excitement and express their creative freedom. ENTPs can be successful in a wide range of careers, as long as they do not feel hemmed in or bored. As debaters with great communication skills, careers in law can offer the challenge and diversity that ENTPs crave. Jobs in the business world that combine the ENTPs rationality, creativity, and natural leadership abilities can also be very rewarding."

            Bh = "Tips for Interacting With ENTPs:"
            h10 = "Friendships"
            p10 = "ENTPs are great at getting along with people no matter what their personality type. While they are usually laid-back, they can be quite competitive. If you are friends with an ENTP, be careful not to get into the habit of trying to out-do each other. Be aware of their love for debates and be careful not to escalate good-natured discussions into combative arguments."
            h11 = "Parenting:"
            p11 = (
                "ENTPs have a fun-loving nature and are excited to share their sense of wonder with their children. Parents with this personality type are supportive, but they have a tendency to try to turn every situation into a learning opportunity.",
                "Parents of ENTP children should be aware that their children may seem argumentative at times, it stems from their natural love for discussion and debate. They may also seem inconsistent at times, being warm and affectionate in one moment and then withdrawing in the next as they become wrapped up in new ideas. Parents should encourage their children to focus on goals and finish the things that they start.")
            h12 = "Relationships"
            p12 = "In intimate relationships, ENTPs can be passionate and exciting. They are warm, loving, and good at understanding their partner's needs. You may find that they may struggle to follow through on promises that they have made, which can be a source of frustration at times. Be aware of your ENTP partners need for spontaneity. You can help balance your partner's impulsiveness by helping them work toward their goals with enthusiasm and practicality."
            h13 = "Popular ENTP Careers"
            p13 = ("Engineer",
                   "Lawyer",
                   "Scientist",
                   "Psychologist",
                   "Inventor",
                   "Psychiatrist",
                   "Journalist")
            h14 = "ENTPs You Might Know"
            p14 = ("Thomas Edison, inventor",
                   "John Adams, U.S. President",
                   "Walt Disney, filmmaker",
                   "Julia Child, cook",
                   "Alexander the Great, King and military leader")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 4:
            h1 = "ESFJ: The Caregiver"
            p1 = "ESFJ stands for (Extraverted, Sensing, Feeling, Judging). People with an ESFJ personality type tend to be outgoing, loyal, organized and tender-hearted. ESFJs gain energy from interacting with other people. They are typically described as outgoing and gregarious. They have a way of encouraging other people to be their best and often have a hard time believing anything bad about the people to whom they are close."
            h2 = "Key ESFJ Characteristics"
            p2 = (
                "In addition to deriving pleasure from helping others, ESFJs also ​have a need for approval. They expect their kind and giving ways to be noticed and appreciated by others. They are sensitive to the needs and feelings of others and are good at responding and providing the care that people need. They want to be liked by others and are easily hurt by unkindness or indifference.",
                "ESFJs derive their value system from external sources including the community at large rather than from intrinsic, ethical, and moral guidelines. People with this personality type who are raised with high values and standards grow up to be generous adults. ESFJs raised in a less enriched environment may have skewed ethics as adults and are more likely to be manipulative and self-centered.",
                "ESFJs also have a strong desire to exert control over their environment. Organizing, planning, and scheduling help people with this personality type feel in command of the world around them.",
                "ESFJs are naturally geared toward understanding other people. They are careful observers of others and are adept at supporting and bringing out the best in people. Because they are so good at helping others feel good about themselves, people feel drawn to ESFJs.",
                "One common myth about ESFJs is that they can sometimes be doormat - allowing others to walk over them out of a fear of disapproval or rejection. While they are people pleasers, this does not mean that they are pushovers.")
            h3 = "Strengths"
            p3 = ("Kind and loyal",
                  "Outgoing",
                  "Organized",
                  "Practical and dependable",
                  "Enjoys helping others")
            h4 = "Weaknesses:"
            p4 = ("Needy",

                  "Approval-seeking",

                  "Sensitive to criticism")
            h5 = "Dominant: Extraverted Feeling"
            p5 = (
                "•  ESFJs tend to make decisions based on personal feeling, emotions, and concern for others. They tend to think more about the personal impact of a decision rather than considering objective criteria. ",
                "•   ESFJs tend to judge people and situations based upon their gut feelings. They often make snap decisions as a result and are quick to share their feelings and opinions. This tendency can be great in some ways, as it allows them to make choices rather quickly. On the negative side, it can sometimes lead to overly harsh judgments of others.")

            h6 = "Auxiliary: Introverted Sensing"
            p6 = "ESFJs are more focused on the present than on the future. They are interested in concrete, immediate details rather than abstract or theoretical information."
            h7 = "Tertiary: Extraverted Intuition"
            p7 = (
                "•	This cognitive function helps ESFJs make connections and find creative solutions to problems.",
                "•	ESFJs are known to explore the possibilities when looking at a situation. They can often find patterns that allow them to gain insights into people and experiences.")
            h8 = "Personal Relationships"
            p8 = (
                "As extroverts, ESFJs love spending time with other people. Not only do they gain energy from social interaction, they are genuinely interested in the well-being of others. They are frequently described as warm-hearted and empathetic, and they will often put the needs of others ahead of their own.",
                "They typically feel insecure in situations where things are uncertain or chaotic. While this makes EFFJs well suited to positions that involve managing or supervising people, it can also lead to conflicts when they try to exercise control over people who do not welcome such direction.")

            h9 = "Career Paths"

            p9 = "Because ESFJs enjoy helping others, they often do well in practical settings that involve taking a caregiver role. Social service and healthcare careers are two areas in particular in which ESFJs may excel at applying their helping nature and desire for order."

            Bh = "Tips for Interacting With ESFJs:"
            h10 = "Friendships"
            p10 = "ESFJ can be selfless to the point of overlooking their own needs to ensure that other people are happy. Being a good friend to someone with this personality type means you should express your appreciation of their giving nature, but also make sure that you reciprocate their kindness."
            h11 = "Parenting:"
            p11 = "ESFJ children are responsible and enjoy spending time with their family. Your child needs regular encouragement. Show your involvement by showing enthusiasm and support for their interests and activities. These children also have a strong need for routine. Establish rules and stick to them. ESFJ children feel more assured and confident when they know what they can expect."
            h12 = "Relationships"
            p12 = "In romance, ESFJs are very devoted, supportive, and loyal. They are not interested in casual relationships and are focused on developing long-term commitments. You can support your partner by showing how much you love and appreciate them. Being responsive by showing affection and returning gestures of love is important."
            h13 = "Popular ESFJ Careers"
            p13 = ("Childcare",
                   "Nursing",
                   "Teaching",
                   "Social work",
                   "Counseling",
                   "Physician",
                   "Recptionist",
                   "Bookkeeper",
                   "Office manager")
            h14 = "ESFJs You Might Know"
            p14 = ("Sally Field, actress",
                   "Sam Walton, Wal-Mart founder",
                   "William McKinley, U.S. President",
                   "Barbara Walters, television journalist",
                   "Joy, film character, Inside Out")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 0:
            h1 = "ENFJ: The Giver"
            p1 = "ENFJ stands for (Extraverted, Intuitive, Feeling, Judging).People with ENFJ personality type are often described as warm, outgoing, loyal, and sensitive."
            h2 = "Key ENFJ Characteristics"
            p2 = (
                "ENFJs are strong extraverts; then sincerely enjoy spending time with other people. They have great people skills and are often described as warm, affectionate and supportive. Not only are people with this personality type great at encouraging other people, they also derive personal satisfaction from helping others.",
                "ENFJs are often so interested in devoting their time to others that they can neglect their own needs. They also have a tendency to be too hard on themselves, blaming themselves for when things go wrong and not giving themselves enough credit when things go right. Because of this, it is important that people with this personality type regularly set aside some time to attend to their own needs.",
                "They are also good at bringing consensus among diverse people. For this reason, they can be outstanding leaders and bring an enthusiasm to a group that can be motivating and inspirational.",
                "One common myth about ENFJs is that they are always sociable. While they love people, they do need time alone in order to assimilate and organize their thoughts.")
            h3 = "Strengths"
            p3 = ("Outgoing and warm-hearted",
                  "Empathetic",
                  "Wide social circle",
                  "Encouraging",
                  "Organized")
            h4 = "Weaknesses:"
            p4 = ("Approval-seeking",

                  "Overly sensitive",

                  "Indecisive",

                  "Self-sacrificing")
            h5 = "Dominant: Extraverted Feeling"
            p5 = (
                "•  ENFJs express this cognitive function through their engaging social behavior and harmonious social relationships. They are in tune with other people's feelings, often to the point that they ignore their own needs in order to please others. ",
                "•  ENFJs place a stronger emphasis on personal, subject considerations rather than objective criteria when making decisions. How a decision will impact others is often a primary concern. ")

            h6 = "Auxiliary: Introverted Intuition"
            p6 = "ENFJs like to think about the future rather than the present. They may often become so focused on the larger goal that they lose sight of the immediate details. As ENFJs take in information about the world, their introverted intuition processes this data in order to create impressions, ideas, and thoughts. This allows them to spot patterns and make sense of complex or abstract data."
            h7 = "Tertiary: Extraverted Sensing"
            p7 = "In an ENFJs personality, extraverted sensing causes them to take in the present moment, gathering concrete details and sensory information from the environment. Because of this, they will often seek out novel or interesting experiences and sensations. People with this personality type tend to be very aware of their present environment. This can lead to a great appreciation of aesthetics and a desire to create a pleasing space."
            h8 = "Personal Relationships"
            p8 = (
                "ENFJs value other people highly and are warm, nurturing, and supportive in personal relationships. At times they can become very wrapped up in other people's problems. They are altruistic and interested in helping others, which can sometimes come off as a bit overbearing. Despite this, they are usually very well liked and people appreciate their genuine concern and care.",
                "As parents, ENFJs are nurturing and warm, although they can sometimes be accused of being so-called helicopter parents. They are directly involved in their children's lives, although they can sometimes be quite strict and even rigid at times. ENFJs need to remember to give their children room to explore and express their individuality, particularly as children age into adolescence.",
                "ENFJs have an outgoing personality and enjoy spending time with other people. Being in social settings helps them feel energized. In friendships and other relationships, people typically describe ENFJs as supportive and fun to be around. They are particularly good at relating to others and are known to help bring out the best in the people with whom they spend their time.")

            h9 = "Career Paths"

            p9 = "ENFJs often do best in careers where they get to help other people and spend a great deal of time interacting with others. Because of their strong communication and organizational skills, ENFJs can make great leaders and managers. They are good at organizing activities, helping each group member achieve their potential and resolving interpersonal conflicts. They strive to create harmony in all situations, and always seem to know what to do to ease tensions and minimize disagreements."

            Bh = "Tips for Interacting With INTJs:"
            h10 = "Friendships"
            p10 = "One of the best ways to be a good friend to an ENFJ is to accept the care and support that they naturally offer. People with this personality type enjoy helping their friends, and it is important to show that you accept and appreciate what they have to offer. However, it is also important that you offer your support in return. ENFJs are not always good at asking for help when they need it. In many cases, simply being willing to listen to whatever they have to share can be very helpful."
            h11 = "Parenting:"
            p11 = (
                "Children of ENFJs might find it difficult to live up to their parents' high exceptions. At times, the ENFJ parent's hands-on approach to parenting can be stifling and make it difficult for kids to explore the world on their own terms.",
                "Parents of ENFJ children should recognize that their children are extremely empathetic, sometimes to the point that they may feel overwhelmed by the strong emotions that other people evoke. These children are giving and caring but may find it difficult to burden others with their own struggles. Parents should encourage their children to care for others, while still taking care of their own emotional well-being.")
            h12 = "Relationships"
            p12 = "Because ENFJs are so sensitive to the feelings of others, your happiness is critical to your partner's happiness. Remember that your partner may even put their own needs last in order to ensure that your needs are met. Let your ENFJ partner know how much you appreciate all the support and care that they offer and be willing to provide the same support in return – even if he or she struggles to ask for help."
            h13 = "Popular ENFJ Careers"
            p13 = ("Counselor",
                   "Teacher",
                   "Psychologist",
                   "Social Worker",
                   "Human Resources Manager",
                   "Sales Representative",
                   "Manager")
            h14 = "ENFJs You Might Know"
            p14 = ("Abraham Maslow, psychologist",
                   "Peyton Manning, football player",
                   "Barack Obama, U.S. president",
                   "Bono, musician",
                   "Elizabeth Bennet, character in Pride and Prejudice")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 2:
            h1 = "ENTJ: The Commander"
            p1 = "ENTJ stands for (Extraverted, Intuitive, Thinking, Judging). Other people often describe people with this type of personality as assertive, confident, and outspoken."
            h2 = "Key ENTJ Characteristics"
            p2 = (
                "People with this personality type enjoy spending time with other people. They have strong verbal skills and interacting with others helps them feel energized.",
                "ENTJ types prefer to think about the future rather than focus on the here-and-now. They usually find abstract and theoretical information more interesting than concrete details.",
                "When making decisions, ENTJs place a greater emphasis on objective and logical information. Personal feeling and the emotions of others tend not to factor much into their choices.",
                "ENTJs are planners. Making decisions and having a schedule or course of action planned out gives them a sense of predictability and control.",
                "They are highly rational, good at spotting problems, and excel at taking charge. These tendencies make them natural leaders who are focused on efficiently solving problems.",
                "One myth about ENTJs is that they are cold and ruthless. While they are not necessarily good with emotions, this does not mean that they are intentionally cruel. They are prone to hiding their own emotions and sentimentality, viewing it as a weakness that should not be made known to others.")
            h3 = "Strengths"
            p3 = ("Strong leadership skills",
                  "Self-assured",
                  "Well-organized",
                  "Good at making decisions",
                  "Assertive and outspoken",
                  "Strong communication skills")
            h4 = "Weaknesses:"
            p4 = ("Impatient",

                  "Stubborn",

                  "Insensitive",

                  "Aggressive",

                  "Intolerant")
            h5 = "Dominant: Extraverted Thinking"
            p5 = (
                "•   This is an ENTJ preferred functioned and is expressed through the way they make decisions and judgments.",
                "•	ENTJs have a tendency to speak first without listening, making snap judgments before really taking in all the information pertaining to a situation.",
                "•  While they tend to make snap judgments, they are also very rational and objective. They are focused on imposing order and standards on the world around them. Setting measurable goals is important. ")

            h6 = "Auxiliary: Introverted Intuition"
            p6 = (
                "• People with this personality type are future-focused and always consider the possibilities when approaching a decision.",
                "• ENTJs are forward-thinking and are not afraid of change. They trust their instincts, although they may have a tendency to regret jumping to conclusions so quickly.")
            h7 = "Tertiary: Extraverted Sensing"
            p7 = (
                "•	This cognitive function gives ENTJs an appetite for adventure. They enjoy novel experiences and may sometimes engage in thrill-seeking behaviors.",
                "•	Because their outward sensory focus, they also have an appreciation for beautiful things in life. They often enjoy surrounding themselves with things that they find attractive or interesting.")
            h8 = "Personal Relationships"
            p8 = (
                "Since ENTJs are extroverts, they gain energy from socializing (unlike introverts, who expend energy in social situations). They love having passionate and lively conversations and debates. In some cases, other people can feel intimidated by the ENTJs confidence and strong verbal skills. When they have a good idea, people with this personality type feel compelled to share their point of view with others.",
                "Despite their verbal abilities, ENTJs are not always good at understanding other people's emotions. Expressing emotions can be difficult for them at times, and their tendency to get into debates can make them seem aggressive, argumentative, and confrontational. People can overcome this problem by making a conscious effort to think about how other people might be feeling.",
                "They may struggle to understand or get along with more sensitive personality types. While they are extraverts, they are not emotionally expressive and other people may see them as insensitive.")

            h9 = "Career Paths"

            p9 = "Thanks to their comfort in the spotlight, ability to communicate, and a tendency to make quick decisions, ENTJs tend to naturally fall into leadership roles."

            Bh = "Tips for Interacting With ENTJs:"
            h10 = "Friendships"
            p10 = "ENTJ are social people and love engaging conversations. While they can seem argumentative and confrontational at times, just remember that this is part of their communication style. Try not to take it personally. They tend to have the easiest friendships with people who share their interests and views, and may struggle to understand people who are very introverted, sensitive, or emotional."
            h11 = "Parenting:"
            p11 = (
                "Parents of ENTJ children should recognize that their child is independent and intellectually curious. You can help your child by allowing them to pursue their curiosity. Understand that your child will often need your reasoning explained in order to understand why certain rules needs to be followed.",
                "You can also help your child develop their emotional understanding by talking openly about feelings. Point out how people might feel about different experiences so that your ENTJ child can learn to better interpret both their own emotions and those of others.")
            h12 = "Relationships"
            p12 = "An ENTJ partner can often seem quite dominating in a relationship. Because dealing with emotions does not come naturally to them, they may seem insensitive to their partner's feelings. It is important to remember that this does not mean that ENTJ’s don’t have feelings — they do need to feel completely comfortable in order to show their emotions. They are very committed to making relationships work and are always looking for ways that they can improve their relationships. If you have an issue with your partner, be upfront and honest. Your partner would rather hear the truth than try to guess your feelings."
            h13 = "Popular ENTJ Careers"
            p13 = ("Human Resources Manager",
                   "Company CEO or manager",
                   "Lawyer",
                   "Scientist",
                   "Software Developer",
                   "Business analyst",
                   "ENtrepreneur",
                   "University Professor")
            h14 = "ENTJs You Might Know"
            p14 = ("Franklin D. Roosevelt, U.S. President",
                   "Bill Gates, Microsoft founder",
                   "Vince Lombardi, football coach",
                   "Carl Sagan, astronomer",
                   "Lex Luthor, Superman character")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        elif result == 6:
            h1 = "ESTJ: The Director"
            p1 = "ESTJ stands for (Extraverted, Sensing, Thinking, Judging). They are assertive and are very concerned with making sure that things run smoothly and according to the rules. They are committed to tradition, standards, and laws. They have strong beliefs and possess sensible judgement, and expect that other people uphold these same principles as well."
            h2 = "Key ESTJ Characteristics"
            p2 = (
                "Individuals with this personality type tend to place a high value on tradition, rules, and security. Maintaining the status quo is important to ESTJs, and they often become involved in civic duties, government branches, and community organizations.",
                "Because of their orthodox approach to life, they can sometimes be seen as rigid, stubborn, and unyielding. Their take-charge attitude makes it easy for ESTJs to assume leadership positions.",
                "Their self-confidence and strong convictions help them excel at putting plans into action, but they can at times appear critical and overly aggressive, particularly when other people fail to live up to their high standards.",
                "People often describe ESTJs as predictable, stable, committed, and practical. They tend to be very frank and honest when it comes to sharing their opinions, which can sometimes be seen as harsh or overly critical.")
            h3 = "Strengths"
            p3 = ("Practical and realistic",
                  "Dependable",
                  "Self-confident",
                  "Hard-working",
                  "Traditional",
                  "Strong leadership skills")
            h4 = "Weaknesses:"
            p4 = ("Insensitive",

                  "Inflexible",

                  "Not good at expressing feelings",

                  "Argumentative",

                  "Bossy")
            h5 = "Dominant: Extraverted Thinking"
            p5 = (
                "•   ESTJs rely on objective information and logic to make decisions rather than personal feelings. They are skilled at making objective, impersonal, and impartial decisions. Rather than focusing on their own subjective feelings when they are making judgments, they consider facts and logic in order to make rational choices.",
                "•	People with ESTJ personality types tend to be very practical. They enjoy learning about things that they can see an immediate, real-world use for, but tend to lose interest in things that are abstract or theoretical. ESTJs enjoy concrete facts as opposed to abstract information.",
                "•   They are good at making fast and decisive choices, but they may often rush to judgment before considering all the information about a situation. One the positive side, this trait makes them good leaders, but it can sometimes lead them to being viewed as harsh or abrasive.")

            h6 = "Auxiliary: Introverted Sensing"
            p6 = (
                "• They are good at remembering things with a great deal of detail. Their memories of past events can be quite vivid, and they often utilize their recollections of past experiences to make connections with present events.",
                "• Because their sensing function is focused inwardly, they tend to be less concerned with novelty and more focused on familiarity. They enjoy having habits and routines that they can depend upon. While this gives them stability and predictability, it can also make them stubborn and unyielding at times.")
            h7 = "Tertiary: Extraverted Intuition"
            p7 = (
                "•	This aspect of personality seeks out novel ideas and possibilities. It compels people with this personality type to explore their creativity.",
                "•	As they process new ideas and information, they may explore the possible meanings in order to spot new connections or patterns. This allows them to look at incoming information and recognize that there may be more than one interpretation or possible outcome.")
            h8 = "Personal Relationships"
            p8 = (
                "As extroverts, ESTJs are very outgoing and enjoy spending time in the company of others. They can be very boisterous and funny in social situations and often enjoy being at the center of attention.",
                "Family is also of the utmost importance to ESTJs. They put a great deal of effort into fulfilling their family obligations. Social events are also important and they are good at remembering important events such as birthdays and anniversaries. They look forward to attending weddings, family reunions, holiday parties, class reunions, and other occasions.",
                "One potential area of weakness for ESTJs is their tendency to be rigid when it comes to rules and routines. They take their own opinion quite seriously, and are often less inclined to listen to what others have to say on a subject.")

            h9 = "Career Paths"

            p9 = "Because they appreciate order and organization, they frequently do well in supervisory roles.When in such positions, they are committed to making sure that members of the group follow rules and traditions and law established by higher authorities."

            Bh = "Tips for Interacting With ESTJs:"
            h10 = "Friendships"
            p10 = "People with this personality type are sociable and enjoy getting their friends involved in activities that they enjoy. ESTJs often value dependability over almost everything else. If you are a stable friend who sticks to your commitments, you will likely be able to forge a strong friendships with an ESTJ."
            h11 = "Parenting:"
            p11 = "ESTJs children tend to be very responsible and goal-directed, but be cautious to avoid placing too many expectations on your child's shoulders. They enjoy structure and routine. While they are good at being self-directed, they still need guidance and rules to give them the security they crave."
            h12 = "Relationships"
            p12 = "ESTJs are dependable and take their commitments seriously. Once they have dedicated themselves to a relationship, they will stay true to it for life. They tend to avoid emotions and feelings, which can be difficult for their partners as times. While they may not express how they feel through words, remember that they will often convey their emotions through actions."
            h13 = "Popular ESTJ Careers"
            p13 = ("Police Officer",
                   "Military",
                   "Judge",
                   "Politician",
                   "Teacher",
                   "School administrator",
                   "Business manager",
                   "Accountant",
                   "Banker")
            h14 = "ESTJs You Might Know"
            p14 = ("Lyndon B. Johnson, U.S. President",
                   "Megyn Kelly, journalist",
                   "Billy Graham, evangelist",
                   "Alec Baldwin, actor",
                   "Darth Vader, character from Star Wars")

            return h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14
        #return show
