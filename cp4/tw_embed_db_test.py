import logging

import psycopg2
import sshtunnel
import numpy as np


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    db_conn = None
    db_cur = None
    try:
        with sshtunnel.open_tunnel(('rivanna.hpc.virginia.edu', 22),
                                   ssh_username='<Your Rivanna User Name>',
                                   ssh_password='<Your Rivanna Password>',
                                   remote_bind_address=('postgis1-s.bii.virginia.edu', 5432),
                                   local_bind_address=('localhost', 1234)) as tunnel:

            tunnel.start()
            print("server connected")

            db_conn = psycopg2.connect(host=tunnel.local_bind_host,
                                    port=tunnel.local_bind_port,
                                    dbname='socialsim',
                                    user='<Your postgis1 User Name>',
                                    password='<Your postgis1 Password>')
            print("database connected")
            db_cur = db_conn.cursor()

            '''Your SQL code starts here'''
            # 1) Find all users who have more than 1 tweets of a specific tweet type within a given time interval
            # TODO
            # Please fill in appropriate values for the following three variables.
            datetime_start = '2019-1-23 18:00:00'
            datetime_end = '2019-1-24 10:12:00'
            tw_type = 'retweet' # ['retweet'|'quote'|'reply'|'tweet']

            # TODO
            # Please modify the 'where' block to add or remove conditions
            sql_str_format = """select
                                    cp4.tw_statuses.usr_id
                                from
	                                cp4.tw_statuses
	                            where
	                                cp4.tw_statuses.action_type = '{0}'
	                                and
                                    cp4.tw_statuses.created_at >= timestamp '{1}'
                                    and 
                                    cp4.tw_statuses.created_at <= timestamp '{2}'
                                group by
	                                cp4.tw_statuses.usr_id
	                            having
	                                count(cp4.tw_statuses.id) > 1
                            """
            db_cur.execute(sql_str_format.format(tw_type, datetime_start, datetime_end))
            l_recs = db_cur.fetchall()
            l_usr_ids = []
            if l_recs is not None:
                l_usr_ids = [rec[0].strip() for rec in l_recs]
                print('User IDs: ')
                print(l_usr_ids)
            else:
                logging.debug('No matching records.')

            # TODO
            # Take advantage of the resulting user IDs.

            # 2) For a given user, a given time interval and a specific embedding type, get the average embedding vector
            # over all embedding vectors satisfying the conditions.
            # TODO
            # Please fill in appropriate values for the following three variables.
            spec_usr_id = '0bTBiIi5y65fPYqahRJy5w'
            datetime_start = '2019-1-23 18:00:00'
            datetime_end = '2019-1-24 10:12:00'
            embed_type = 'avg_nps_vect'

            sql_str_format = """select 
                                    cp4.tw_statuses.id, 
                                    string_to_array(cp4.tw_embeddings.{0}, ',')::real[]
                                from 
                                    cp4.tw_statuses 
                                    inner join 
                                    cp4.tw_embeddings 
                                    on cp4.tw_statuses.id = cp4.tw_embeddings.id
                                where 
                                    cp4.tw_statuses.usr_id = '{1}' 
                                    and 
                                    cp4.tw_statuses.created_at >= timestamp '{2}'
                                    and 
                                    cp4.tw_statuses.created_at <= timestamp '{3}'
                                    """
            db_cur.execute(sql_str_format.format(embed_type, spec_usr_id, datetime_start, datetime_end))
            l_recs = db_cur.fetchall()
            avg_embed = np.zeros(300)
            embed_cnt = 0
            if l_recs is not None:
                for idx, rec in enumerate(l_recs):
                    tw_id = rec[0]
                    embed = rec[1]
                    print('Rec %s: ' % idx)
                    print(tw_id)
                    print(embed)
                    avg_embed += embed
                    embed_cnt += 1
                if embed_cnt > 0:
                    avg_embed = avg_embed / embed_cnt
                print('\nUser:%s, Datetime Start:%s, Datetime End:%s, Embed Type:%s, Avg Embed:'
                      % (spec_usr_id, datetime_start, datetime_end, embed_type))
                print(avg_embed)
            else:
                logging.debug('No matching record.')

            # TODO
            # Take advantage of the resulting average embedding vector, 'avg_embed'.

            '''Your SQL code ends here'''

            tunnel.stop()
    except Exception as err:
        print("Connection Failed %s" % err)
    finally:
        try:
            if db_cur is not None:
                db_cur.close()
                logging.debug('DB cursor has been closed.')
            if db_conn is not None:
                db_conn.close()
                logging.debug('DB connection has been closed.')
        except:
            pass