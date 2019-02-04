import tensorflow as tf


def summarize(summary_scalars, session, writer, cnt, values):
  with tf.variable_scope("statistics"):
    summary_placeholders, summary_ops = {}, {}
    for tag in summary_scalars:
      summary_placeholders[tag] = tf.placeholder('float32', None)
      summary_ops[tag] = tf.summary.scalar(tag, summary_placeholders[tag])
  ops = [summary_ops[tag] for tag in list(values.keys())]
  feed_dict = {summary_placeholders[tag]: values[tag] for tag in list(values.keys())}
  summary_lists = session.run(ops, feed_dict)
  for summary in summary_lists:
    writer.add_summary(summary, cnt)
